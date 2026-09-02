#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compDEM 4.5.1 — comparaison robuste de deux DEM photogrammétriques."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.windows import Window, from_bounds

__version__ = "4.5.1"

# Profil V4_GOLDEN validé. Les paramètres métier restent internes au code.
PROFILE = {
    "density_window_cm": 5.0,
    "density_core_min_area_cm2": 4.8,
    "density_bridge_radius_cm": 1.6,
    "min_support_area_cm2": 20.0,
    "irregular_min_short_cm": 3.0,
    "irregular_min_long_cm": 5.0,
    "zone_growth_radius_cm": 2.0,
    "line_seed_closing_cm": 1.0,
    "linear_min_width_cm": 2.0,
    "linear_min_length_cm": 20.0,
    "thin_linear_min_width_cm": 1.5,
    "thin_linear_min_length_cm": 30.0,
    "sparse_line_min_median_mm": 150.0,
    "line_corridor_half_width_cm": 2.0,
    "line_max_gap_cm": 40.0,
    "line_projection_bin_cm": 0.4,
    "sign_neighborhood_cm": 5.0,
    "max_opposite_signal_ratio": 0.10,
    "merge_distance_cm": 4.0,
    "spatial_max_min_area_cm2": 2.0,
}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("reference_dem", "compare_dem"):
        if key not in raw:
            raise ValueError(f"Paramètre JSON manquant : {key}")

    def resolve(value: str | Path) -> Path:
        p = Path(value)
        return p.resolve() if p.is_absolute() else (path.parent / p).resolve()

    output_dir = resolve(raw.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "reference_dem": resolve(raw["reference_dem"]),
        "compare_dem": resolve(raw["compare_dem"]),
        "threshold_mm": float(raw.get("threshold_mm", 10.0)),
        "output_prefix": str(raw.get("output_prefix", "result")),
        "output_dir": output_dir,
    }


def output_paths(cfg: dict) -> dict[str, Path]:
    d, p = cfg["output_dir"], cfg["output_prefix"]
    return {
        "detections": d / f"{p}_detections.geojson",
        "zones": d / f"{p}_zones.geojson",
        "boxes": d / f"{p}_boxes.geojson",
        "summary": d / f"{p}_summary.json",
        "difference": d / f"{p}_difference.tif",
        "rgba": d / f"{p}_difference_rgba.tif",
    }


def _integer_window(window: Window, width: int, height: int) -> Window:
    c0 = max(0, min(width, int(round(window.col_off))))
    r0 = max(0, min(height, int(round(window.row_off))))
    c1 = max(0, min(width, int(round(window.col_off + window.width))))
    r1 = max(0, min(height, int(round(window.row_off + window.height))))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("Intersection DEM vide.")
    return Window(c0, r0, c1 - c0, r1 - r0)


def read_common_area(cfg: dict):
    """Retourne diff = compare-reference sur l'intersection monde, sans rééchantillonnage."""
    with rasterio.open(cfg["reference_dem"]) as ref, rasterio.open(cfg["compare_dem"]) as cmp:
        left, bottom = max(ref.bounds.left, cmp.bounds.left), max(ref.bounds.bottom, cmp.bounds.bottom)
        right, top = min(ref.bounds.right, cmp.bounds.right), min(ref.bounds.top, cmp.bounds.top)
        if right <= left or top <= bottom:
            raise ValueError("Les deux DEM ne se recouvrent pas.")

        rw = _integer_window(from_bounds(left, bottom, right, top, ref.transform), ref.width, ref.height)
        cw = _integer_window(from_bounds(left, bottom, right, top, cmp.transform), cmp.width, cmp.height)
        a = ref.read(1, window=rw).astype(np.float32)
        b = cmp.read(1, window=cw).astype(np.float32)
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]

        valid = np.isfinite(a) & np.isfinite(b)
        if ref.nodata is not None:
            valid &= a != ref.nodata
        if cmp.nodata is not None:
            valid &= b != cmp.nodata

        diff = (b.astype(np.float64) - a.astype(np.float64)).astype(np.float32)
        diff[~valid] = np.nan
        return diff, valid, ref.window_transform(rw), ref.crs or cmp.crs


def pixel_size_m(transform: Affine) -> float:
    return (math.hypot(transform.a, transform.d) + math.hypot(transform.b, transform.e)) / 2.0


def cm_to_px(cm: float, px_m: float, minimum: int = 1) -> int:
    return max(minimum, int(round((cm / 100.0) / px_m)))


def min_area_dimensions(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > 30000:
        pts = pts[::max(1, len(pts) // 30000)]
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return min(w, h) + 1.0, max(w, h) + 1.0


def opposite_ratio(diff, valid, sign, bbox, margin_px, threshold_m) -> float:
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0 - margin_px), max(0, y0 - margin_px)
    x1, y1 = min(diff.shape[1], x1 + margin_px), min(diff.shape[0], y1 + margin_px)
    local, ok = diff[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    same = ok & ((local >= threshold_m) if sign > 0 else (local <= -threshold_m))
    opp = ok & ((local <= -threshold_m) if sign > 0 else (local >= threshold_m))
    return int(opp.sum()) / max(int(same.sum()), 1)


def world_ring(bbox, transform: Affine):
    x0, y0, x1, y1 = bbox
    pts = [transform * p for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
    return [[float(x), float(y)] for x, y in pts]


def world_axis_aligned_ring(bbox, transform: Affine):
    ring = world_ring(bbox, transform)
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    return [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]


def detect_density_zones(diff, valid, positive, negative, transform, threshold_m):
    px_m = pixel_size_m(transform)
    px_area_cm2 = px_m * px_m * 10000.0
    window_px = cm_to_px(PROFILE["density_window_cm"], px_m)
    if window_px % 2 == 0:
        window_px += 1
    core_min = math.ceil(PROFILE["density_core_min_area_cm2"] / px_area_cm2)
    support_min = math.ceil(PROFILE["min_support_area_cm2"] / px_area_cm2)
    bridge_px = cm_to_px(PROFILE["density_bridge_radius_cm"], px_m)
    growth_px = cm_to_px(PROFILE["zone_growth_radius_cm"], px_m)
    sign_margin = cm_to_px(PROFILE["sign_neighborhood_cm"], px_m)
    min_short = cm_to_px(PROFILE["irregular_min_short_cm"], px_m)
    min_long = cm_to_px(PROFILE["irregular_min_long_cm"], px_m)
    max_opp = PROFILE["max_opposite_signal_ratio"]
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge_px + 1,) * 2)
    growth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * growth_px + 1,) * 2)
    out = []

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
            ratio = opposite_ratio(diff, valid, sign, bbox, sign_margin, threshold_m)
            if ratio > max_opp:
                continue

            local_labels = grown_labels[ys, xs]
            values, counts = np.unique(local_labels[local_labels > 0], return_counts=True)
            if len(values):
                chosen = int(values[np.argmax(counts)])
                gys, gxs = np.where(raw & (grown_labels == chosen))
            else:
                gxs, gys = xs, ys
            vals = diff[gys, gxs]
            out.append({
                "type": "density_irregular", "sign": sign,
                "bbox_px": (int(gxs.min()), int(gys.min()), int(gxs.max()) + 1, int(gys.max()) + 1),
                "support_pixels": int(len(gxs)), "support_area_cm2": float(len(gxs) * px_area_cm2),
                "median_mm": float(np.median(vals) * 1000.0), "mean_mm": float(np.mean(vals) * 1000.0),
                "opposite_ratio": float(ratio), "short_cm": float(short_px * px_m * 100.0),
                "long_cm": float(long_px * px_m * 100.0),
                "_support_xs": gxs.astype(np.int32), "_support_ys": gys.astype(np.int32),
            })
    return out


def find_line_seeds(diff, valid, positive, negative, transform, threshold_m):
    px_m = pixel_size_m(transform)
    support_min = math.ceil(PROFILE["min_support_area_cm2"] / (px_m * px_m * 10000.0))
    margin = cm_to_px(PROFILE["sign_neighborhood_cm"], px_m)
    normal_w, normal_l = cm_to_px(PROFILE["linear_min_width_cm"], px_m), cm_to_px(PROFILE["linear_min_length_cm"], px_m)
    thin_w, thin_l = cm_to_px(PROFILE["thin_linear_min_width_cm"], px_m), cm_to_px(PROFILE["thin_linear_min_length_cm"], px_m)
    k = cm_to_px(PROFILE["line_seed_closing_cm"], px_m)
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
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
            if not ((short_px >= normal_w and long_px >= normal_l) or (short_px >= thin_w and long_px >= thin_l)):
                continue
            bbox = (int(sx.min()), int(sy.min()), int(sx.max()) + 1, int(sy.max()) + 1)
            if opposite_ratio(diff, valid, sign, bbox, margin, threshold_m) > PROFILE["max_opposite_signal_ratio"]:
                continue
            if float(np.median(np.abs(diff[sy, sx])) * 1000.0) < PROFILE["sparse_line_min_median_mm"]:
                continue
            seeds.append({"sign": sign, "seed_xs": sx, "seed_ys": sy})
    return seeds


def grow_line_pca(seed, diff, positive_points, negative_points, transform):
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
    t, p = relative @ axis, relative @ perpendicular
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
    groups, start, previous = [], int(occupied[0]), int(occupied[0])
    for current in occupied[1:]:
        current = int(current)
        if current - previous <= max_gap_bins:
            previous = current
        else:
            groups.append((start, previous))
            start = previous = current
    groups.append((start, previous))
    seed_bin = (seed_mid - t0) / bin_px

    def distance(group):
        a, b = group
        return 0.0 if a <= seed_bin <= b else min(abs(seed_bin - a), abs(seed_bin - b))

    a, b = min(groups, key=distance)
    keep = in_corridor & (t >= t0 + a * bin_px) & (t <= t0 + (b + 1) * bin_px)
    kept = points[keep]
    if len(kept) == 0:
        return None
    kx, ky = kept[:, 0].astype(np.int64), kept[:, 1].astype(np.int64)
    vals = diff[ky, kx]
    return {
        "type": "sparse_line_pca", "sign": sign,
        "bbox_px": (int(kx.min()), int(ky.min()), int(kx.max()) + 1, int(ky.max()) + 1),
        "support_pixels": int(len(kx)), "median_mm": float(np.median(vals) * 1000.0),
        "mean_mm": float(np.mean(vals) * 1000.0), "median_abs_mm": float(np.median(np.abs(vals)) * 1000.0),
        "_support_xs": kx.astype(np.int32), "_support_ys": ky.astype(np.int32),
    }


def boxes_close(a, b, d: int) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + d < bx0 or bx1 + d < ax0 or ay1 + d < by0 or by1 + d < ay0)


def merge_boxes(items, distance_px: int, same_sign_only: bool):
    if not items:
        return []
    used, merged = [False] * len(items), []
    for i in range(len(items)):
        if used[i]:
            continue
        group, used[i], changed = [i], True, True
        while changed:
            changed = False
            bbox = (
                min(items[k]["bbox_px"][0] for k in group), min(items[k]["bbox_px"][1] for k in group),
                max(items[k]["bbox_px"][2] for k in group), max(items[k]["bbox_px"][3] for k in group),
            )
            signs = {items[k]["sign"] for k in group}
            for j in range(len(items)):
                if used[j] or (same_sign_only and items[j]["sign"] not in signs):
                    continue
                if boxes_close(bbox, items[j]["bbox_px"], distance_px):
                    used[j], changed = True, True
                    group.append(j)
        bbox = (
            min(items[k]["bbox_px"][0] for k in group), min(items[k]["bbox_px"][1] for k in group),
            max(items[k]["bbox_px"][2] for k in group), max(items[k]["bbox_px"][3] for k in group),
        )
        signs = sorted({items[k]["sign"] for k in group})
        merged.append({
            "type": "merged_zone" if same_sign_only else "final_box",
            "sign": signs[0] if len(signs) == 1 else 0, "signs": signs,
            "bbox_px": bbox, "members": [items[k] for k in group],
        })
    return merged


def detect_all(diff, valid, transform, threshold_mm: float):
    threshold_m = threshold_mm / 1000.0
    positive = valid & (diff >= threshold_m)
    negative = valid & (diff <= -threshold_m)
    candidates = detect_density_zones(diff, valid, positive, negative, transform, threshold_m)
    py, px = np.where(positive)
    ny, nx = np.where(negative)
    positive_points = np.column_stack([px, py]).astype(np.float64)
    negative_points = np.column_stack([nx, ny]).astype(np.float64)
    for seed in find_line_seeds(diff, valid, positive, negative, transform, threshold_m):
        grown = grow_line_pca(seed, diff, positive_points, negative_points, transform)
        if grown is not None:
            candidates.append(grown)
    zones = merge_boxes(candidates, cm_to_px(PROFILE["merge_distance_cm"], pixel_size_m(transform)), True)
    return candidates, zones, merge_boxes(zones, 0, False)


def sign_name(sign: int) -> str:
    return "positive" if sign > 0 else "negative" if sign < 0 else "mixed"


def save_geojson(path: Path, features: list, crs, properties: dict):
    data = {"type": "FeatureCollection", "properties": properties, "features": features}
    if crs is not None:
        data["crs"] = {"type": "name", "properties": {"name": crs.to_wkt()}}

    # Conserver une présentation décimale stable dans le texte GeoJSON tout en
    # gardant de vrais nombres JSON (et non des chaînes).
    text = json.dumps(data, ensure_ascii=False, indent=2)
    fixed_decimals = {
        "bbox_width_m": 4,
        "bbox_height_m": 4,
        "median_depth_mm": 1,
        "spatial_max_depth_mm": 1,
    }
    for key, decimals in fixed_decimals.items():
        pattern = re.compile(rf'("{re.escape(key)}"\s*:\s*)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
        text = pattern.sub(lambda m, d=decimals: m.group(1) + f"{float(m.group(2)):.{d}f}", text)

    path.write_text(text, encoding="utf-8")


def write_tfw(tiff_path: Path, transform: Affine):
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
    threshold_m = threshold_mm / 1000.0
    positive = valid & np.isfinite(diff) & (diff > threshold_m)
    negative = valid & np.isfinite(diff) & (diff < -threshold_m)
    rgba = np.zeros((4, *diff.shape), dtype=np.uint8)
    rgba[0, positive], rgba[2, negative], rgba[3, positive | negative] = 255, 255, 255
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
    members = item.get("members")
    if members is None:
        yield item
    else:
        for member in members:
            yield from iter_leaf_detections(member)


def detected_support_for_box(box_item, diff):
    """Renvoie les pixels détectés uniques d'une box : x, y et valeur dz."""
    width, ids = diff.shape[1], []
    for det in iter_leaf_detections(box_item):
        xs, ys = det.get("_support_xs"), det.get("_support_ys")
        if xs is not None and ys is not None and len(xs):
            ids.append(ys.astype(np.int64) * width + xs.astype(np.int64))
    if not ids:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)
    unique_ids = np.unique(np.concatenate(ids))
    ys, xs = unique_ids // width, unique_ids % width
    values = diff[ys, xs]
    finite = np.isfinite(values)
    return xs[finite], ys[finite], values[finite]


def spatial_confirmed_max_depth_mm(xs, ys, values, transform, min_area_cm2=2.0):
    """Maximum |dz| confirmé par une composante 8-connexe d'au moins min_area_cm2.

    Les pixels positifs et négatifs sont traités séparément : ils ne peuvent donc
    jamais s'additionner artificiellement pour atteindre la surface minimale.
    Une recherche binaire permet de trouver le seuil de profondeur maximal sans
    tester toutes les valeurs une par une.
    """
    if not len(values):
        return None

    px_m = pixel_size_m(transform)
    px_area_cm2 = px_m * px_m * 10000.0
    min_pixels = max(1, int(math.ceil(min_area_cm2 / px_area_cm2)))
    confirmed_depths = []

    for sign in (1, -1):
        select = values > 0 if sign > 0 else values < 0
        if int(np.sum(select)) < min_pixels:
            continue

        sx, sy = xs[select], ys[select]
        depths = np.abs(values[select]).astype(np.float32)
        x0, x1 = int(sx.min()), int(sx.max()) + 1
        y0, y1 = int(sy.min()), int(sy.max()) + 1
        lx, ly = sx - x0, sy - y0
        levels = np.unique(depths)

        def has_confirmed_component(level):
            active = depths >= level
            if int(np.sum(active)) < min_pixels:
                return False
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            mask[ly[active], lx[active]] = 1
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            return n > 1 and int(stats[1:, cv2.CC_STAT_AREA].max()) >= min_pixels

        # Propriété monotone : quand le seuil descend, le nombre de pixels actifs
        # ne peut qu'augmenter. On cherche donc le plus haut niveau encore confirmé.
        lo, hi, best = 0, len(levels) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_confirmed_component(levels[mid]):
                best = float(levels[mid])
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            # Conserver le signe du changement : positif = gain, négatif = perte.
            confirmed_depths.append(sign * best * 1000.0)

    # Si une box contient exceptionnellement les deux signes, conserver le
    # maximum spatial confirmé ayant la plus grande amplitude absolue.
    return max(confirmed_depths, key=abs) if confirmed_depths else None


def box_depth_stats(box_item, diff, transform):
    """Statistiques métier calculées uniquement sur les pixels réellement détectés."""
    xs, ys, values = detected_support_for_box(box_item, diff)
    if not len(values):
        return {"detected_pixel_count": 0, "median_depth_mm": None,
                "spatial_max_depth_mm": None}

    # Médiane signée sur TOUS les pixels détectés. Aucun filtrage P99.
    median_depth_mm = float(np.median(values) * 1000.0)
    spatial_max = spatial_confirmed_max_depth_mm(
        xs, ys, values, transform, min_area_cm2=PROFILE["spatial_max_min_area_cm2"]
    )
    return {
        "detected_pixel_count": int(len(values)),
        "median_depth_mm": round(median_depth_mm, 1),
        "spatial_max_depth_mm": None if spatial_max is None else round(float(spatial_max), 1),
    }


def export_results(candidates, zones, final_boxes, diff, valid, transform, crs, cfg):
    paths, threshold_mm = output_paths(cfg), cfg["threshold_mm"]
    common = {"algorithm": "V4.5 signed spatial stats on V4_GOLDEN geometry",
              "difference": "compare - reference", "threshold_mm": threshold_mm,
              "spatial_max_min_area_cm2": PROFILE["spatial_max_min_area_cm2"]}

    detections = []
    for i, det in enumerate(candidates, 1):
        props = {"detection_id": i, "type": det["type"], "sign": sign_name(det["sign"]),
                 "threshold_mm": threshold_mm, "support_pixels": int(det.get("support_pixels", 0))}
        for key in ("support_area_cm2", "median_mm", "mean_mm", "median_abs_mm", "opposite_ratio", "short_cm", "long_cm"):
            if key in det:
                props[key] = round(float(det[key]), 4)
        detections.append({"type": "Feature", "properties": props,
                           "geometry": {"type": "Polygon", "coordinates": [world_ring(det["bbox_px"], transform)]}})

    zone_features = [{
        "type": "Feature",
        "properties": {"zone_id": i, "sign": sign_name(z["sign"]), "member_count": len(z["members"]),
                       "threshold_mm": threshold_mm},
        "geometry": {"type": "Polygon", "coordinates": [world_ring(z["bbox_px"], transform)]},
    } for i, z in enumerate(zones, 1)]

    box_features = []
    for i, b in enumerate(final_boxes, 1):
        ring = world_axis_aligned_ring(b["bbox_px"], transform)
        props = {
            "box_id": i,
            "zone_count": len(b["members"]), "threshold_mm": threshold_mm,
            "bbox_width_m": round(ring[1][0] - ring[0][0], 4),
            "bbox_height_m": round(ring[2][1] - ring[1][1], 4),
            **box_depth_stats(b, diff, transform),
        }
        box_features.append({"type": "Feature", "properties": props,
                             "geometry": {"type": "Polygon", "coordinates": [ring]}})

    save_geojson(paths["detections"], detections, crs, common)
    save_geojson(paths["zones"], zone_features, crs, common)
    save_geojson(paths["boxes"], box_features, crs, common)
    save_difference_tiff(paths["difference"], diff, transform, crs)
    save_rgba_tiff(paths["rgba"], diff, valid, transform, crs, threshold_mm)

    finite = diff[np.isfinite(diff)]
    summary = {
        **common, "intersection_width_px": int(diff.shape[1]), "intersection_height_px": int(diff.shape[0]),
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


def main():
    parser = argparse.ArgumentParser(description="Détection V4.5 de changements entre deux DEM")
    parser.add_argument("config", help="Fichier JSON de configuration")
    args = parser.parse_args()
    cfg = load_config(args.config)
    diff, valid, transform, crs = read_common_area(cfg)
    candidates, zones, boxes = detect_all(diff, valid, transform, cfg["threshold_mm"])
    print(json.dumps(export_results(candidates, zones, boxes, diff, valid, transform, crs, cfg),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
