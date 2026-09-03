#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compDEM 4.5.6 — sorties métier inversées sur le moteur V4_GOLDEN 4.5.3.

Le moteur de détection reste inchangé dans compdem_core.py.
Sorties: reference - compare ; gain positif ; loss négatif.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import compdem_core as core

__version__ = "4.5.6"


def save_geojson(path: Path, features: list, crs, properties: dict):
    data = {"type": "FeatureCollection", "properties": properties, "features": features}
    if crs is not None:
        data["crs"] = {"type": "name", "properties": {"name": crs.to_wkt()}}
    text = json.dumps(data, ensure_ascii=False, indent=2)
    fixed_decimals = {
        "bbox_width_m": 3,
        "bbox_height_m": 3,
        "center_y_m": 3,
        "center_angle_deg": 1,
        "detected_area_m2": 4,
        "median_depth_mm": 1,
        "spatial_max_depth_mm": 1,
    }
    for key, decimals in fixed_decimals.items():
        pattern = re.compile(rf'("{re.escape(key)}"\s*:\s*)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
        text = pattern.sub(lambda m, d=decimals: m.group(1) + f"{float(m.group(2)):.{d}f}", text)
    path.write_text(text, encoding="utf-8")


def box_stats(box_item, diff, transform):
    """Stats sur l'union des vrais pixels détectés, avec signe affiché inversé."""
    xs, ys, values = core.detected_support_for_box(box_item, diff)
    if not len(values):
        return {
            "detected_pixel_count": 0,
            "detected_area_m2": 0.0,
            "median_depth_mm": None,
            "spatial_max_depth_mm": None,
        }

    median_mm = -float(np.median(values) * 1000.0)
    spatial_internal = core.spatial_confirmed_max_depth_mm(
        xs,
        ys,
        values,
        transform,
        min_area_cm2=core.PROFILE["spatial_max_min_area_cm2"],
    )
    spatial_mm = None if spatial_internal is None else -float(spatial_internal)
    pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)

    return {
        "detected_pixel_count": int(len(values)),
        "detected_area_m2": round(float(len(values) * pixel_area_m2), 4),
        "median_depth_mm": round(median_mm, 1),
        "spatial_max_depth_mm": None if spatial_mm is None else round(spatial_mm, 1),
    }


def change_type(value_mm):
    if value_mm is None:
        return None
    return "gain" if value_mm >= 0 else "loss"


def export_results(candidates, zones, final_boxes, diff, valid, transform, crs, cfg):
    paths = core.output_paths(cfg)
    threshold_mm = cfg["threshold_mm"]
    display_diff = -diff

    common = {
        "algorithm": "V4.5.6 inverted display convention on V4_GOLDEN geometry",
        "difference": "reference - compare",
        "change_convention": "gain = positive, loss = negative",
        "threshold_mm": threshold_mm,
        "spatial_max_min_area_cm2": core.PROFILE["spatial_max_min_area_cm2"],
    }

    detections = []
    for i, det in enumerate(candidates, 1):
        props = {
            "detection_id": i,
            "type": det["type"],
            "sign": core.sign_name(-det["sign"]),
            "threshold_mm": threshold_mm,
            "support_pixels": int(det.get("support_pixels", 0)),
        }
        for key in ("support_area_cm2", "median_abs_mm", "opposite_ratio", "short_cm", "long_cm"):
            if key in det:
                props[key] = round(float(det[key]), 4)
        for key in ("median_mm", "mean_mm"):
            if key in det:
                props[key] = round(-float(det[key]), 4)
        detections.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [core.world_ring(det["bbox_px"], transform)]},
        })

    zone_features = [{
        "type": "Feature",
        "properties": {
            "zone_id": i,
            "sign": core.sign_name(-z["sign"]),
            "member_count": len(z["members"]),
            "threshold_mm": threshold_mm,
        },
        "geometry": {"type": "Polygon", "coordinates": [core.world_ring(z["bbox_px"], transform)]},
    } for i, z in enumerate(zones, 1)]

    corners = [
        transform * (0, 0),
        transform * (diff.shape[1], 0),
        transform * (0, diff.shape[0]),
        transform * (diff.shape[1], diff.shape[0]),
    ]
    world_left = min(x for x, _ in corners)
    world_right = max(x for x, _ in corners)
    world_width = world_right - world_left

    box_features = []
    for i, box in enumerate(final_boxes, 1):
        ring = core.world_axis_aligned_ring(box["bbox_px"], transform)
        minx, maxx = ring[0][0], ring[1][0]
        miny, maxy = ring[0][1], ring[2][1]
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0
        angle = 0.0 if world_width == 0 else (center_x - world_left) / world_width * 360.0
        stats = box_stats(box, diff, transform)
        sign_value = stats["median_depth_mm"]
        if sign_value is None or sign_value == 0:
            sign_value = stats["spatial_max_depth_mm"]

        props = {
            "box_id": i,
            "zone_count": len(box["members"]),
            "threshold_mm": threshold_mm,
            "bbox_width_m": round(maxx - minx, 3),
            "bbox_height_m": round(maxy - miny, 3),
            "center_y_m": round(center_y, 3),
            "center_angle_deg": round(angle, 1),
            "change_type": change_type(sign_value),
            **stats,
        }
        box_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    save_geojson(paths["detections"], detections, crs, common)
    save_geojson(paths["zones"], zone_features, crs, common)
    save_geojson(paths["boxes"], box_features, crs, common)

    # COG numérique : signe inversé. COG couleur : on garde l'image historique
    # du moteur (ancien négatif=bleu), qui correspond désormais à gain positif=bleu.
    core.save_difference_tiff(paths["difference"], display_diff, transform, crs)
    core.save_rgba_tiff(paths["rgba"], diff, valid, transform, crs, threshold_mm)

    finite = display_diff[np.isfinite(display_diff)]
    summary = {
        **common,
        "intersection_width_px": int(diff.shape[1]),
        "intersection_height_px": int(diff.shape[0]),
        "valid_pixels": int(len(finite)),
        "pixels_above_positive_threshold": int(np.sum(finite >= threshold_mm / 1000.0)),
        "pixels_below_negative_threshold": int(np.sum(finite <= -threshold_mm / 1000.0)),
        "candidate_count": len(candidates),
        "zone_count": len(zones),
        "box_count": len(final_boxes),
        "density_candidate_count": sum(c["type"] == "density_irregular" for c in candidates),
        "sparse_line_candidate_count": sum(c["type"] == "sparse_line_pca" for c in candidates),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Détection V4.5.6 de changements entre deux DEM")
    parser.add_argument("config", help="Fichier JSON de configuration")
    args = parser.parse_args()

    cfg = core.load_config(args.config)
    diff, valid, transform, crs = core.read_common_area(cfg)
    candidates, zones, boxes = core.detect_all(diff, valid, transform, cfg["threshold_mm"])
    print(json.dumps(export_results(candidates, zones, boxes, diff, valid, transform, crs, cfg),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
