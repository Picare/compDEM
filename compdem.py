#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compDEM - V4.2 STATS

Refactoring de V4_GOLDEN : même algorithme métier, interface simplifiée.

Entrée : un petit JSON contenant les deux DEM, le seuil et le préfixe de sortie.
Sorties :
  - *_detections.geojson
  - *_zones.geojson
  - *_boxes.geojson
  - *_summary.json
  - *_difference.tif + .tfw
  - *_difference_rgba.tif + .tfw

Les paramètres calibrés de V4_GOLDEN sont volontairement internes au code.
Ils ne doivent pas être modifiés par un utilisateur normal.

Dépendances :
    pip install numpy rasterio opencv-python

Utilisation :
    python compdem.py config.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.windows import Window, from_bounds

__version__ = "4.2.0"


# =============================================================================
# 1. PROFIL V4_GOLDEN
# =============================================================================
#
# Ces valeurs ont été calibrées sur les jeux de référence validés.
# Elles sont volontairement cachées du JSON utilisateur : ce sont des détails
# internes de l'algorithme, pas des réglages métier courants.
#
PROFILE = {
    # Zones compactes / irrégulières
    "density_window_cm": 5.0,
    "density_core_min_area_cm2": 4.8,
    "density_bridge_radius_cm": 1.6,
    "min_support_area_cm2": 20.0,
    "irregular_min_short_cm": 3.0,
    "irregular_min_long_cm": 5.0,
    "zone_growth_radius_cm": 2.0,

    # Lignes fortes / fragmentées
    "line_seed_closing_cm": 1.0,   # 5 px sur les jeux à 2 mm/px
    "linear_min_width_cm": 2.0,
    "linear_min_length_cm": 20.0,
    "thin_linear_min_width_cm": 1.5,
    "thin_linear_min_length_cm": 30.0,
    "sparse_line_min_median_mm": 150.0,
    "line_corridor_half_width_cm": 2.0,
    "line_max_gap_cm": 40.0,
    "line_projection_bin_cm": 0.4,

    # Rejet des zones où positif et négatif sont trop mélangés
    "sign_neighborhood_cm": 5.0,
    "max_opposite_signal_ratio": 0.10,

    # Regroupement final des fragments du même signe
    "merge_distance_cm": 4.0,
}


# =============================================================================
# 2. CONFIGURATION UTILISATEUR
# =============================================================================

def load_config(path: str | Path) -> dict:
    """Lit le JSON minimal et prépare automatiquement tous les chemins de sortie."""
    config_path = Path(path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    for key in ("reference_dem", "compare_dem"):
        if key not in cfg:
            raise ValueError(f"Paramètre JSON manquant : {key}")

    base = config_path.parent

    def resolve(value: str | Path) -> Path:
        p = Path(value)
        return p.resolve() if p.is_absolute() else (base / p).resolve()

    reference = resolve(cfg["reference_dem"])
    compare = resolve(cfg["compare_dem"])
    threshold_mm = float(cfg.get("threshold_mm", 10.0))
    prefix = str(cfg.get("output_prefix", "result"))
    output_dir = resolve(cfg.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "reference_dem": reference,
        "compare_dem": compare,
        "threshold_mm": threshold_mm,
        "output_prefix": prefix,
        "output_dir": output_dir,
    }


def output_paths(cfg: dict) -> dict[str, Path]:
    """Construit les noms de fichiers automatiquement à partir du préfixe."""
    d = cfg["output_dir"]
    p = cfg["output_prefix"]
    return {
        "detections": d / f"{p}_detections.geojson",
        "zones": d / f"{p}_zones.geojson",
        "boxes": d / f"{p}_boxes.geojson",
        "summary": d / f"{p}_summary.json",
        "difference": d / f"{p}_difference.tif",
        "rgba": d / f"{p}_difference_rgba.tif",
    }


# =============================================================================
# 3. LECTURE DES DEUX DEM SUR LEUR INTERSECTION MONDE
# =============================================================================

def _integer_window(window: Window, width: int, height: int) -> Window:
    """Arrondit une fenêtre Rasterio sur la grille de pixels et la borne au raster."""
    c0 = max(0, min(width, int(round(window.col_off))))
    r0 = max(0, min(height, int(round(window.row_off))))
    c1 = max(0, min(width, int(round(window.col_off + window.width))))
    r1 = max(0, min(height, int(round(window.row_off + window.height))))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("Intersection DEM vide.")
    return Window(c0, r0, c1 - c0, r1 - r0)


def read_common_area(cfg: dict):
    """
    Lit uniquement la zone commune aux deux DEM et calcule :

        différence = DEM_comparé - DEM_référence

    Rasterio lit automatiquement le TFW placé à côté du TIFF lorsqu'il existe.
    Aucun rééchantillonnage ni aucune interpolation n'est effectué.
    """
    with rasterio.open(cfg["reference_dem"]) as ref, rasterio.open(cfg["compare_dem"]) as cmp:
        # Intersection des emprises dans le repère monde.
        left = max(ref.bounds.left, cmp.bounds.left)
        bottom = max(ref.bounds.bottom, cmp.bounds.bottom)
        right = min(ref.bounds.right, cmp.bounds.right)
        top = min(ref.bounds.top, cmp.bounds.top)
        if right <= left or top <= bottom:
            raise ValueError("Les deux DEM ne se recouvrent pas.")

        ref_window = _integer_window(from_bounds(left, bottom, right, top, ref.transform), ref.width, ref.height)
        cmp_window = _integer_window(from_bounds(left, bottom, right, top, cmp.transform), cmp.width, cmp.height)

        ref_data = ref.read(1, window=ref_window).astype(np.float32)
        cmp_data = cmp.read(1, window=cmp_window).astype(np.float32)

        # Tolérance aux rasters ayant une ligne/colonne de différence.
        h = min(ref_data.shape[0], cmp_data.shape[0])
        w = min(ref_data.shape[1], cmp_data.shape[1])
        ref_data = ref_data[:h, :w]
        cmp_data = cmp_data[:h, :w]

        valid = np.isfinite(ref_data) & np.isfinite(cmp_data)
        if ref.nodata is not None:
            valid &= ref_data != ref.nodata
        if cmp.nodata is not None:
            valid &= cmp_data != cmp.nodata

        diff = (cmp_data.astype(np.float64) - ref_data.astype(np.float64)).astype(np.float32)
        diff[~valid] = np.nan

        transform = ref.window_transform(ref_window)
        crs = ref.crs or cmp.crs

    return diff, valid, transform, crs


# =============================================================================
# 4. OUTILS PIXELS / GÉOMÉTRIE
# =============================================================================

def pixel_size_m(transform: Affine) -> float:
    """Taille moyenne d'un pixel en mètres."""
    px = math.hypot(transform.a, transform.d)
    py = math.hypot(transform.b, transform.e)
    return (px + py) / 2.0


def cm_to_px(cm: float, px_m: float, minimum: int = 1) -> int:
    return max(minimum, int(round((cm / 100.0) / px_m)))


def min_area_dimensions(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Dimensions du plus petit rectangle orienté autour des pixels."""
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > 30000:
        pts = pts[::max(1, len(pts) // 30000)]
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return min(w, h) + 1.0, max(w, h) + 1.0


def opposite_ratio(diff, valid, sign, bbox, margin_px, threshold_m) -> float:
    """Part de signal de signe opposé autour d'une détection."""
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - margin_px), max(0, y0 - margin_px)
    x1 = min(diff.shape[1], x1 + margin_px)
    y1 = min(diff.shape[0], y1 + margin_px)
    local = diff[y0:y1, x0:x1]
    ok = valid[y0:y1, x0:x1]
    if sign > 0:
        same = ok & (local >= threshold_m)
        opp = ok & (local <= -threshold_m)
    else:
        same = ok & (local <= -threshold_m)
        opp = ok & (local >= threshold_m)
    return int(opp.sum()) / max(int(same.sum()), 1)


def world_ring(bbox_px, transform: Affine) -> list[list[float]]:
    """Convertit une box pixel en anneau GeoJSON dans les coordonnées monde."""
    x0, y0, x1, y1 = bbox_px
    pts = [
        transform * (x0, y0),
        transform * (x1, y0),
        transform * (x1, y1),
        transform * (x0, y1),
        transform * (x0, y0),
    ]
    return [[float(x), float(y)] for x, y in pts]


def world_axis_aligned_ring(bbox_px, transform: Affine) -> list[list[float]]:
    """Bounding box X/Y monde finale."""
    ring = world_ring(bbox_px, transform)
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]


# =============================================================================
# 5. DÉTECTION DES ZONES COMPACTES / IRRÉGULIÈRES
# =============================================================================

def detect_density_zones(diff, valid, positive, negative, transform, threshold_m):
    """Détecte des noyaux denses puis récupère les fragments voisins du même signe."""
    px_m = pixel_size_m(transform)
    px_area_cm2 = px_m * px_m * 10000.0

    window_px = cm_to_px(PROFILE["density_window_cm"], px_m)
    if window_px % 2 == 0:
        window_px += 1
    core_min = math.ceil(PROFILE["density_core_min_area_cm2"] / px_area_cm2)
    support_min = math.ceil(PROFILE["min_support_area_cm2"] / px_area_cm2)
    bridge_px = cm_to_px(PROFILE["density_bridge_radius_cm"], px_m)
    growth_px = cm_to_px(PROFILE["zone_growth_radius_cm"], px_m)
    sign_margin_px = cm_to_px(PROFILE["sign_neighborhood_cm"], px_m)
    min_short = cm_to_px(PROFILE["irregular_min_short_cm"], px_m)
    min_long = cm_to_px(PROFILE["irregular_min_long_cm"], px_m)
    max_opp = PROFILE["max_opposite_signal_ratio"]

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge_px + 1,) * 2)
    growth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * growth_px + 1,) * 2)
    detections = []

    for sign, raw, opposite in ((1, positive, negative), (-1, negative, positive)):
        same_count = cv2.boxFilter(raw.astype(np.float32), -1, (window_px, window_px), normalize=False,
                                   borderType=cv2.BORDER_CONSTANT)
        opp_count = cv2.boxFilter(opposite.astype(np.float32), -1, (window_px, window_px), normalize=False,
                                  borderType=cv2.BORDER_CONSTANT)
        core = (same_count >= core_min) & (opp_count <= max_opp * np.maximum(same_count, 1.0)) & valid

        bridged = cv2.dilate(raw.astype(np.uint8), bridge_kernel)
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
        labels_with_core = np.unique(labels[core])

        grown_bridge = cv2.dilate(raw.astype(np.uint8), growth_kernel)
        _, grown_labels, _, _ = cv2.connectedComponentsWithStats(grown_bridge, connectivity=8)

        for label in labels_with_core:
            if label == 0 or label >= n_labels:
                continue
            ys, xs = np.where(raw & (labels == label))
            if len(xs) < support_min:
                continue
            short_px, long_px = min_area_dimensions(xs, ys)
            if short_px < min_short or long_px < min_long:
                continue

            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            ratio = opposite_ratio(diff, valid, sign, bbox, sign_margin_px, threshold_m)
            if ratio > max_opp:
                continue

            # Croissance vers les fragments proches du même signe.
            g_labels = grown_labels[ys, xs]
            values, counts = np.unique(g_labels[g_labels > 0], return_counts=True)
            if len(values):
                chosen = int(values[np.argmax(counts)])
                gys, gxs = np.where(raw & (grown_labels == chosen))
            else:
                gxs, gys = xs, ys

            vals = diff[gys, gxs]
            detections.append({
                "type": "density_irregular",
                "sign": sign,
                "bbox_px": (int(gxs.min()), int(gys.min()), int(gxs.max()) + 1, int(gys.max()) + 1),
                "support_pixels": int(len(gxs)),
                "support_area_cm2": float(len(gxs) * px_area_cm2),
                "median_mm": float(np.median(vals) * 1000.0),
                "mean_mm": float(np.mean(vals) * 1000.0),
                "opposite_ratio": float(ratio),
                "short_cm": float(short_px * px_m * 100.0),
                "long_cm": float(long_px * px_m * 100.0),
                # Coordonnées exactes des pixels réellement retenus par la détection.
                # Elles ne sont pas exportées dans le GeoJSON, mais servent à calculer
                # les statistiques des boxes sans inclure le fond du rectangle.
                "_support_xs": gxs.astype(np.int32),
                "_support_ys": gys.astype(np.int32),
            })

    return detections


# =============================================================================
# 6. DÉTECTION DES LIGNES FORTES ET FRAGMENTÉES
# =============================================================================

def find_line_seeds(diff, valid, positive, negative, transform, threshold_m):
    """Sélectionne les fragments assez fiables pour amorcer une croissance PCA."""
    px_m = pixel_size_m(transform)
    px_area_cm2 = px_m * px_m * 10000.0
    support_min = math.ceil(PROFILE["min_support_area_cm2"] / px_area_cm2)
    sign_margin_px = cm_to_px(PROFILE["sign_neighborhood_cm"], px_m)

    normal_w = cm_to_px(PROFILE["linear_min_width_cm"], px_m)
    normal_l = cm_to_px(PROFILE["linear_min_length_cm"], px_m)
    thin_w = cm_to_px(PROFILE["thin_linear_min_width_cm"], px_m)
    thin_l = cm_to_px(PROFILE["thin_linear_min_length_cm"], px_m)

    kernel_size = cm_to_px(PROFILE["line_seed_closing_cm"], px_m)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    seeds = []
    for sign, raw in ((1, positive), (-1, negative)):
        closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        closed &= valid.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

        for label in range(1, n):
            if int(stats[label, cv2.CC_STAT_AREA]) < 20:
                continue
            x, y = int(stats[label, 0]), int(stats[label, 1])
            w, h = int(stats[label, 2]), int(stats[label, 3])
            component = labels[y:y+h, x:x+w] == label
            support = component & raw[y:y+h, x:x+w]
            sy, sx = np.where(support)
            if len(sx) < support_min:
                continue
            sx, sy = sx + x, sy + y
            short_px, long_px = min_area_dimensions(sx, sy)
            if not ((short_px >= normal_w and long_px >= normal_l) or
                    (short_px >= thin_w and long_px >= thin_l)):
                continue

            bbox = (int(sx.min()), int(sy.min()), int(sx.max()) + 1, int(sy.max()) + 1)
            if opposite_ratio(diff, valid, sign, bbox, sign_margin_px, threshold_m) > PROFILE["max_opposite_signal_ratio"]:
                continue
            median_abs_mm = float(np.median(np.abs(diff[sy, sx])) * 1000.0)
            if median_abs_mm < PROFILE["sparse_line_min_median_mm"]:
                continue
            seeds.append({"sign": sign, "seed_xs": sx, "seed_ys": sy})

    return seeds


def grow_line_pca(seed, diff, valid, positive_points, negative_points, transform):
    """Prolonge une graine uniquement dans un corridor autour de son axe PCA."""
    px_m = pixel_size_m(transform)
    half_width = cm_to_px(PROFILE["line_corridor_half_width_cm"], px_m)
    max_gap = cm_to_px(PROFILE["line_max_gap_cm"], px_m)
    bin_px = cm_to_px(PROFILE["line_projection_bin_cm"], px_m)

    sign = seed["sign"]
    points = positive_points if sign > 0 else negative_points
    if len(points) == 0:
        return None

    seed_points = np.column_stack([seed["seed_xs"], seed["seed_ys"]]).astype(np.float64)
    center = seed_points.mean(axis=0)
    centered = seed_points - center
    _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    axis = eigenvectors[:, -1]
    perpendicular = np.array([-axis[1], axis[0]])

    relative = points - center
    t = relative @ axis
    p = relative @ perpendicular
    in_corridor = np.abs(p) <= half_width
    if not np.any(in_corridor):
        return None

    corridor_t = t[in_corridor]
    seed_t = centered @ axis
    seed_mid = (float(seed_t.min()) + float(seed_t.max())) / 2.0
    t0 = math.floor(float(corridor_t.min()) / bin_px) * bin_px
    ids = np.floor((corridor_t - t0) / bin_px).astype(np.int64)
    occupied = np.where(np.bincount(ids) > 0)[0]
    if len(occupied) == 0:
        return None

    max_gap_bins = max(1, math.ceil(max_gap / bin_px))
    groups = []
    start = previous = int(occupied[0])
    for current in occupied[1:]:
        current = int(current)
        if current - previous <= max_gap_bins:
            previous = current
        else:
            groups.append((start, previous))
            start = previous = current
    groups.append((start, previous))

    seed_bin = (seed_mid - t0) / bin_px
    def group_distance(g):
        a, b = g
        return 0.0 if a <= seed_bin <= b else min(abs(seed_bin - a), abs(seed_bin - b))

    a, b = min(groups, key=group_distance)
    lower_t, upper_t = t0 + a * bin_px, t0 + (b + 1) * bin_px
    keep = in_corridor & (t >= lower_t) & (t <= upper_t)
    kept = points[keep]
    if len(kept) == 0:
        return None

    kx, ky = kept[:, 0].astype(np.int64), kept[:, 1].astype(np.int64)
    vals = diff[ky, kx]
    return {
        "type": "sparse_line_pca",
        "sign": sign,
        "bbox_px": (int(kx.min()), int(ky.min()), int(kx.max()) + 1, int(ky.max()) + 1),
        "support_pixels": int(len(kx)),
        "median_mm": float(np.median(vals) * 1000.0),
        "mean_mm": float(np.mean(vals) * 1000.0),
        "median_abs_mm": float(np.median(np.abs(vals)) * 1000.0),
        "_support_xs": kx.astype(np.int32),
        "_support_ys": ky.astype(np.int32),
    }


# =============================================================================
# 7. REGROUPEMENT DES DÉTECTIONS ET SUPPRESSION DES BOXES SUPERPOSÉES
# =============================================================================

def boxes_close(a, b, d: int) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + d < bx0 or bx1 + d < ax0 or ay1 + d < by0 or by1 + d < ay0)


def merge_boxes(items, distance_px: int, same_sign_only: bool):
    """Fusion transitive des boxes proches ; la dernière passe interdit les chevauchements."""
    if not items:
        return []
    used = [False] * len(items)
    merged = []

    for i in range(len(items)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            bbox = (
                min(items[k]["bbox_px"][0] for k in group),
                min(items[k]["bbox_px"][1] for k in group),
                max(items[k]["bbox_px"][2] for k in group),
                max(items[k]["bbox_px"][3] for k in group),
            )
            signs = {items[k]["sign"] for k in group}
            for j in range(len(items)):
                if used[j] or (same_sign_only and items[j]["sign"] not in signs):
                    continue
                if boxes_close(bbox, items[j]["bbox_px"], distance_px):
                    used[j] = True
                    group.append(j)
                    changed = True

        bbox = (
            min(items[k]["bbox_px"][0] for k in group),
            min(items[k]["bbox_px"][1] for k in group),
            max(items[k]["bbox_px"][2] for k in group),
            max(items[k]["bbox_px"][3] for k in group),
        )
        signs = sorted({items[k]["sign"] for k in group})
        merged.append({
            "type": "merged_zone" if same_sign_only else "final_box",
            "sign": signs[0] if len(signs) == 1 else 0,
            "signs": signs,
            "bbox_px": bbox,
            "members": [items[k] for k in group],
        })
    return merged


def detect_all(diff, valid, transform, threshold_mm: float):
    """Pipeline V4_GOLDEN : densité + lignes PCA + deux passes de fusion."""
    threshold_m = threshold_mm / 1000.0
    positive = valid & (diff >= threshold_m)
    negative = valid & (diff <= -threshold_m)

    candidates = detect_density_zones(diff, valid, positive, negative, transform, threshold_m)

    # Coordonnées calculées une seule fois puis réutilisées par toutes les graines PCA.
    py, px = np.where(positive)
    ny, nx = np.where(negative)
    positive_points = np.column_stack([px, py]).astype(np.float64)
    negative_points = np.column_stack([nx, ny]).astype(np.float64)

    for seed in find_line_seeds(diff, valid, positive, negative, transform, threshold_m):
        grown = grow_line_pca(seed, diff, valid, positive_points, negative_points, transform)
        if grown is not None:
            candidates.append(grown)

    merge_px = cm_to_px(PROFILE["merge_distance_cm"], pixel_size_m(transform))
    zones = merge_boxes(candidates, merge_px, same_sign_only=True)
    final_boxes = merge_boxes(zones, 0, same_sign_only=False)
    return candidates, zones, final_boxes


# =============================================================================
# 8. EXPORTS GEOJSON + TIFF + TFW
# =============================================================================

def sign_name(sign: int) -> str:
    return "positive" if sign > 0 else "negative" if sign < 0 else "mixed"


def save_geojson(path: Path, features: list, crs, properties: dict):
    data = {"type": "FeatureCollection", "properties": properties, "features": features}
    if crs is not None:
        data["crs"] = {"type": "name", "properties": {"name": crs.to_wkt()}}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tfw(tiff_path: Path, transform: Affine):
    """Écrit le world-file au centre du pixel supérieur gauche."""
    cx = transform.c + 0.5 * transform.a + 0.5 * transform.b
    cy = transform.f + 0.5 * transform.d + 0.5 * transform.e
    values = [transform.a, transform.d, transform.b, transform.e, cx, cy]
    tiff_path.with_suffix(".tfw").write_text("\n".join(f"{v:.15f}" for v in values) + "\n", encoding="utf-8")


def save_difference_tiff(path: Path, diff, transform, crs):
    profile = {
        "driver": "GTiff", "height": diff.shape[0], "width": diff.shape[1], "count": 1,
        "dtype": "float32", "transform": transform, "crs": crs, "compress": "DEFLATE",
        "predictor": 3, "tiled": True, "blockxsize": 256, "blockysize": 256, "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(diff.astype(np.float32), 1)
    write_tfw(path, transform)


def save_rgba_tiff(path: Path, diff, valid, transform, crs, threshold_mm: float):
    """Transparent entre -seuil et +seuil ; rouge au-dessus, bleu en dessous."""
    threshold_m = threshold_mm / 1000.0
    positive = valid & np.isfinite(diff) & (diff > threshold_m)
    negative = valid & np.isfinite(diff) & (diff < -threshold_m)
    rgba = np.zeros((4, *diff.shape), dtype=np.uint8)
    rgba[0, positive] = 255
    rgba[2, negative] = 255
    rgba[3, positive | negative] = 255

    profile = {
        "driver": "GTiff", "height": diff.shape[0], "width": diff.shape[1], "count": 4,
        "dtype": "uint8", "transform": transform, "crs": crs, "compress": "DEFLATE",
        "tiled": True, "blockxsize": 256, "blockysize": 256, "photometric": "RGB",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(rgba)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    write_tfw(path, transform)


def iter_leaf_detections(item):
    """Parcourt récursivement une box/zone et renvoie ses détections élémentaires.

    Une box finale contient des zones, et chaque zone contient une ou plusieurs
    détections. Cette fonction permet de retrouver les vrais pixels détectés,
    indépendamment de la taille géométrique de la bounding box.
    """
    members = item.get("members")
    if members is None:
        yield item
        return
    for member in members:
        yield from iter_leaf_detections(member)


def detected_values_for_box(box_item, diff):
    """Valeurs DEM uniquement sur l'union des pixels des détections de la box.

    Les mêmes pixels peuvent appartenir à la fois à une détection par densité et
    à une détection de ligne PCA. On déduplique donc les coordonnées avant de
    calculer médiane et maximum. Aucun pixel du simple fond rectangulaire n'est pris.
    """
    width = diff.shape[1]
    ids = []
    for det in iter_leaf_detections(box_item):
        xs = det.get("_support_xs")
        ys = det.get("_support_ys")
        if xs is None or ys is None or len(xs) == 0:
            continue
        ids.append(ys.astype(np.int64) * width + xs.astype(np.int64))

    if not ids:
        return np.empty(0, dtype=np.float32)

    unique_ids = np.unique(np.concatenate(ids))
    ys = unique_ids // width
    xs = unique_ids % width
    values = diff[ys, xs]
    return values[np.isfinite(values)]


def export_results(candidates, zones, final_boxes, diff, valid, transform, crs, cfg):
    paths = output_paths(cfg)
    threshold_mm = cfg["threshold_mm"]
    common = {
        "algorithm": "V4.2 stats on V4.1 / V4_GOLDEN geometry",
        "difference": "compare - reference",
        "threshold_mm": threshold_mm,
    }

    detections = []
    for i, det in enumerate(candidates, 1):
        props = {
            "detection_id": i, "type": det["type"], "sign": sign_name(det["sign"]),
            "threshold_mm": threshold_mm, "support_pixels": int(det.get("support_pixels", 0)),
        }
        for key in ("support_area_cm2", "median_mm", "mean_mm", "median_abs_mm",
                    "opposite_ratio", "short_cm", "long_cm"):
            if key in det:
                props[key] = round(float(det[key]), 4)
        detections.append({
            "type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [world_ring(det["bbox_px"], transform)]},
        })

    zone_features = [{
        "type": "Feature",
        "properties": {
            "zone_id": i, "sign": sign_name(z["sign"]), "member_count": len(z["members"]),
            "threshold_mm": threshold_mm,
        },
        "geometry": {"type": "Polygon", "coordinates": [world_ring(z["bbox_px"], transform)]},
    } for i, z in enumerate(zones, 1)]

    box_features = []
    for i, b in enumerate(final_boxes, 1):
        ring = world_axis_aligned_ring(b["bbox_px"], transform)
        width_cm = (ring[1][0] - ring[0][0]) * 100.0
        height_cm = (ring[2][1] - ring[1][1]) * 100.0

        # IMPORTANT : statistiques uniquement sur les pixels des détections avérées.
        # On ne calcule jamais la médiane sur toute la surface rectangulaire de la box.
        detected_values = detected_values_for_box(b, diff)
        if len(detected_values):
            abs_values_mm = np.abs(detected_values) * 1000.0
            median_depth_mm = float(np.median(abs_values_mm))
            max_depth_mm = float(np.max(abs_values_mm))
            median_dz_mm = float(np.median(detected_values) * 1000.0)
            detected_pixel_count = int(len(detected_values))
        else:
            median_depth_mm = max_depth_mm = median_dz_mm = None
            detected_pixel_count = 0

        box_features.append({
            "type": "Feature",
            "properties": {
                "box_id": i, "sign": sign_name(b["sign"]),
                "signs": [sign_name(s) for s in b.get("signs", [])],
                "zone_count": len(b["members"]), "threshold_mm": threshold_mm,
                "bbox_width_cm": round(width_cm, 2), "bbox_height_cm": round(height_cm, 2),
                "detected_pixel_count": detected_pixel_count,
                # Profondeurs = amplitude absolue des pixels réellement détectés.
                "median_depth_mm": None if median_depth_mm is None else round(median_depth_mm, 3),
                "max_depth_mm": None if max_depth_mm is None else round(max_depth_mm, 3),
                # Valeur signée utile pour savoir si la box correspond plutôt à un gain
                # ou à une perte de matière.
                "median_dz_mm": None if median_dz_mm is None else round(median_dz_mm, 3),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    save_geojson(paths["detections"], detections, crs, common)
    save_geojson(paths["zones"], zone_features, crs, common)
    save_geojson(paths["boxes"], box_features, crs, common)
    save_difference_tiff(paths["difference"], diff, transform, crs)
    save_rgba_tiff(paths["rgba"], diff, valid, transform, crs, threshold_mm)

    finite = diff[np.isfinite(diff)]
    summary = {
        **common,
        "intersection_width_px": int(diff.shape[1]),
        "intersection_height_px": int(diff.shape[0]),
        "valid_pixels": int(len(finite)),
        "pixels_above_positive_threshold": int(np.sum(finite >= threshold_mm / 1000.0)),
        "pixels_below_negative_threshold": int(np.sum(finite <= -threshold_mm / 1000.0)),
        "candidate_count": len(candidates), "zone_count": len(zones), "box_count": len(final_boxes),
        "density_candidate_count": sum(c["type"] == "density_irregular" for c in candidates),
        "sparse_line_candidate_count": sum(c["type"] == "sparse_line_pca" for c in candidates),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Détection V4.2 de changements entre deux DEM")
    parser.add_argument("config", help="Fichier JSON de configuration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    diff, valid, transform, crs = read_common_area(cfg)
    candidates, zones, boxes = detect_all(diff, valid, transform, cfg["threshold_mm"])
    summary = export_results(candidates, zones, boxes, diff, valid, transform, crs, cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
