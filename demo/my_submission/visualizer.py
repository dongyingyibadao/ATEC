"""调试可视化输出——将抓取管线中间结果写入图片和JSON。"""
import json
import math
import os

import cv2
import numpy as np

from .config import IMAGE_HEIGHT, IMAGE_WIDTH
from .camera_transform import FX, FY, CX, CY, project_base_to_pixel


def export_debug_frame(
    debug_dir: str,
    plan_name: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    candidates: list[dict],
    solved: list[dict],
    camera_pose: dict,
    points: dict | None = None,
    object_pose_debug: dict | None = None,
) -> None:
    """导出一帧完整的调试信息到磁盘。"""
    plan_dir = os.path.join(debug_dir, plan_name)
    os.makedirs(plan_dir, exist_ok=True)

    overlay = rgb.copy()
    for idx, cand in enumerate(candidates[:10]):
        color = (0, 255, 255) if idx else (0, 0, 255)
        _draw_grasp_line(overlay, cand["pixel"], cand["angle"], color, str(idx))
        _draw_center_marker(overlay, cand["pixel"], color, f"{cand.get('object_type', '?')}:{idx}")
        _draw_projected_marker(
            overlay,
            project_base_to_pixel(cand.get("object_center_base"), camera_pose),
            (255, 128, 0),
            f"c{idx}",
        )
        _draw_projected_marker(
            overlay,
            project_base_to_pixel(cand.get("grasp_anchor_base"), camera_pose),
            color,
            f"ga{idx}",
        )
    solved_overlay = rgb.copy()
    for idx, cand in enumerate(solved[:5]):
        color = (0, 255, 0) if idx == 0 else (255, 255, 0)
        _draw_grasp_line(solved_overlay, cand["pixel"], cand["angle"], color, f"s{idx}")
        _draw_center_marker(solved_overlay, cand["pixel"], color, f"{cand.get('object_type', '?')}:s{idx}")
        _draw_projected_marker(
            solved_overlay,
            project_base_to_pixel(cand.get("object_center_base"), camera_pose),
            (255, 128, 0),
            f"c{idx}",
        )
        _draw_projected_marker(
            solved_overlay,
            project_base_to_pixel(cand.get("grasp_base"), camera_pose),
            color,
            f"g{idx}",
        )
        _draw_projected_marker(
            solved_overlay,
            project_base_to_pixel(cand.get("pregrasp_base"), camera_pose),
            (255, 0, 255),
            f"p{idx}",
        )

    cv2.imwrite(os.path.join(plan_dir, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(plan_dir, "candidates.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(plan_dir, "solved.png"), cv2.cvtColor(solved_overlay, cv2.COLOR_RGB2BGR))

    if object_pose_debug:
        table_mask = object_pose_debug.get("table_mask")
        if table_mask is not None:
            table_overlay = rgb.copy()
            table_overlay[table_mask.astype(bool)] = (
                0.6 * table_overlay[table_mask.astype(bool)] + 0.4 * np.array([0, 255, 255], dtype=np.float32)
            ).astype(np.uint8)
            cv2.imwrite(os.path.join(plan_dir, "object_pose_table_overlay.png"), cv2.cvtColor(table_overlay, cv2.COLOR_RGB2BGR))
        for strip_name, strip_mask in object_pose_debug.get("strip_masks", {}).items():
            strip_overlay = rgb.copy()
            strip_overlay[strip_mask.astype(bool)] = (
                0.6 * strip_overlay[strip_mask.astype(bool)] + 0.4 * np.array([255, 128, 0], dtype=np.float32)
            ).astype(np.uint8)
            cv2.imwrite(os.path.join(plan_dir, f"strip_{strip_name}_overlay.png"), cv2.cvtColor(strip_overlay, cv2.COLOR_RGB2BGR))
        for band_name, band_mask in object_pose_debug.get("bands", {}).items():
            band_overlay = rgb.copy()
            band_overlay[band_mask.astype(bool)] = (
                0.6 * band_overlay[band_mask.astype(bool)] + 0.4 * np.array([255, 255, 0], dtype=np.float32)
            ).astype(np.uint8)
            cv2.imwrite(os.path.join(plan_dir, f"band_{band_name}_overlay.png"), cv2.cvtColor(band_overlay, cv2.COLOR_RGB2BGR))
        for object_name, components in object_pose_debug.get("component_masks", {}).items():
            comp_overlay = rgb.copy()
            selected_label = (object_pose_debug.get("selected_component_label") or {}).get(object_name)
            for idx, comp in enumerate(components):
                comp_mask = np.asarray(comp.get("mask"), dtype=bool)
                if not np.any(comp_mask):
                    continue
                label = int(comp.get("label", idx + 1))
                color = (
                    int((53 * (idx + 1)) % 255),
                    int((97 * (idx + 2)) % 255),
                    int((181 * (idx + 3)) % 255),
                )
                if selected_label == label:
                    color = (0, 0, 255)
                comp_overlay[comp_mask] = (
                    0.55 * comp_overlay[comp_mask] + 0.45 * np.array(color, dtype=np.float32)
                ).astype(np.uint8)
                ys, xs = np.where(comp_mask)
                if ys.size:
                    cx = int(np.median(xs))
                    cy = int(np.median(ys))
                    cv2.putText(comp_overlay, f"{object_name[0]}c{label}", (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(plan_dir, f"components_{object_name}_overlay.png"), cv2.cvtColor(comp_overlay, cv2.COLOR_RGB2BGR))

    pipeline_debug = {}
    if points is not None:
        pts_base = points["points_base"]
        valid_pts = pts_base[mask]
        pipeline_debug["depth_to_base"] = {
            "total_pixels": int(pts_base.shape[0] * pts_base.shape[1]),
            "valid_pixels": int(valid_pts.shape[0]),
            "points_base_min": valid_pts.min(axis=0).tolist() if valid_pts.size > 0 else [],
            "points_base_max": valid_pts.max(axis=0).tolist() if valid_pts.size > 0 else [],
            "points_base_median": np.median(valid_pts, axis=0).tolist() if valid_pts.size > 0 else [],
        }
        np.save(os.path.join(plan_dir, "points_base_valid.npy"), valid_pts.astype(np.float32))

    pipeline_debug["candidates_grasp_pipeline"] = []
    for c in candidates:
        entry = {
            "object_type": c.get("object_type"),
            "cluster_centroid_base": np.asarray(c.get("cluster_centroid_base", [])).tolist(),
            "icp_model_center_base": np.asarray(c.get("icp_model_center_base", [])).tolist(),
            "surface_base": np.asarray(c.get("surface_base", [])).tolist(),
            "grasp_anchor_base": np.asarray(c.get("grasp_anchor_base", [])).tolist(),
        }
        pipeline_debug["candidates_grasp_pipeline"].append(entry)

    with open(os.path.join(plan_dir, "grasp_pipeline_debug.json"), "w", encoding="utf-8") as fh:
        json.dump(pipeline_debug, fh, indent=2)

    payload = {
        "camera_pose_base": {
            "position_base": np.asarray(camera_pose["position_base"]).tolist(),
            "quat_wxyz": np.asarray(camera_pose["quat_wxyz"]).tolist(),
        },
        "object_pose_debug": {
            "table_mask_pixels": int(np.count_nonzero(object_pose_debug["table_mask"])) if object_pose_debug else 0,
            "strip_mask_pixels": {
                name: int(np.count_nonzero(m))
                for name, m in (object_pose_debug.get("strip_masks", {}) if object_pose_debug else {}).items()
            },
            "band_mask_pixels": {
                name: int(np.count_nonzero(m))
                for name, m in (object_pose_debug.get("bands", {}) if object_pose_debug else {}).items()
            },
            "base_stats": object_pose_debug.get("base_stats", {}) if object_pose_debug else {},
            "component_stats": object_pose_debug.get("component_stats", {}) if object_pose_debug else {},
            "selected_component_label": object_pose_debug.get("selected_component_label", {}) if object_pose_debug else {},
        },
        "candidates": [
            {
                "pixel": c["pixel"],
                "object_type": c.get("object_type"),
                "candidate_source": c.get("candidate_source"),
                "estimated_pose_yaw_rad": c.get("estimated_pose_yaw_rad"),
                "quality": c["quality"],
                "angle_rad": c["angle"],
                "depth_m": c["depth_m"],
                "point_cam_ros": np.asarray(c["point_cam_ros"]).tolist(),
                "point_cam_frame": np.asarray(c.get("point_cam_frame", c["point_cam_ros"])).tolist(),
                "object_center_base": np.asarray(c.get("object_center_base", [])).tolist(),
                "object_center_pixel": project_base_to_pixel(c.get("object_center_base"), camera_pose),
                "surface_base": np.asarray(c["surface_base"]).tolist(),
                "grasp_anchor_base": np.asarray(c.get("grasp_anchor_base", [])).tolist(),
                "grasp_anchor_pixel": project_base_to_pixel(c.get("grasp_anchor_base"), camera_pose),
                "mesh_object_center_base": np.asarray(c.get("mesh_object_center_base", [])).tolist(),
                "mesh_object_rotation": np.asarray(c.get("mesh_object_rotation", [])).tolist(),
                "rank_debug": {
                    "status": c.get("rank_debug", {}).get("status"),
                    "template": c.get("rank_debug", {}).get("template"),
                    "raw_angle_rad": c.get("rank_debug", {}).get("raw_angle_rad"),
                    "template_angle_rad": c.get("rank_debug", {}).get("template_angle_rad"),
                    "surface_base": np.asarray(c.get("rank_debug", {}).get("surface_base", [])).tolist(),
                    "pregrasp_base": np.asarray(c.get("rank_debug", {}).get("pregrasp_base", [])).tolist(),
                    "grasp_base": np.asarray(c.get("rank_debug", {}).get("grasp_base", [])).tolist(),
                    "lift_base": np.asarray(c.get("rank_debug", {}).get("lift_base", [])).tolist(),
                    "pregrasp_result": {
                        "success": c.get("rank_debug", {}).get("pregrasp_result", {}).get("success"),
                        "reason": c.get("rank_debug", {}).get("pregrasp_result", {}).get("reason"),
                        "cost": c.get("rank_debug", {}).get("pregrasp_result", {}).get("cost"),
                        "pos_error": c.get("rank_debug", {}).get("pregrasp_result", {}).get("pos_error"),
                    },
                    "grasp_result": {
                        "success": c.get("rank_debug", {}).get("grasp_result", {}).get("success"),
                        "reason": c.get("rank_debug", {}).get("grasp_result", {}).get("reason"),
                        "cost": c.get("rank_debug", {}).get("grasp_result", {}).get("cost"),
                        "pos_error": c.get("rank_debug", {}).get("grasp_result", {}).get("pos_error"),
                    },
                    "lift_failure_index": c.get("rank_debug", {}).get("lift_failure_index"),
                    "lift_failure_result": {
                        "success": c.get("rank_debug", {}).get("lift_failure_result", {}).get("success"),
                        "reason": c.get("rank_debug", {}).get("lift_failure_result", {}).get("reason"),
                        "cost": c.get("rank_debug", {}).get("lift_failure_result", {}).get("cost"),
                        "pos_error": c.get("rank_debug", {}).get("lift_failure_result", {}).get("pos_error"),
                    },
                },
            }
            for c in candidates
        ],
        "solved": [
            {
                "pixel": c["pixel"],
                "object_type": c.get("object_type"),
                "template_angle_rad": c.get("template_angle_rad"),
                "score": c["score"],
                "object_center_base": np.asarray(c.get("object_center_base", [])).tolist(),
                "object_center_pixel": project_base_to_pixel(c.get("object_center_base"), camera_pose),
                "pregrasp_base": np.asarray(c["pregrasp_base"]).tolist(),
                "pregrasp_pixel": project_base_to_pixel(c.get("pregrasp_base"), camera_pose),
                "grasp_base": np.asarray(c["grasp_base"]).tolist(),
                "grasp_pixel": project_base_to_pixel(c.get("grasp_base"), camera_pose),
                "lift_base": np.asarray(c["lift_base"]).tolist(),
                "lift_bases": [np.asarray(v).tolist() for v in c.get("lift_bases", [])],
            }
            for c in solved
        ],
    }
    with open(os.path.join(plan_dir, "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _draw_grasp_line(image: np.ndarray, pixel: list[int], angle: float, color: tuple, label: str) -> None:
    x, y = int(pixel[0]), int(pixel[1])
    half = 16
    dx = int(math.cos(angle) * half)
    dy = int(math.sin(angle) * half)
    cv2.line(image, (x - dx, y - dy), (x + dx, y + dy), color, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _draw_center_marker(image: np.ndarray, pixel: list[int], color: tuple, label: str) -> None:
    x, y = int(pixel[0]), int(pixel[1])
    cv2.circle(image, (x, y), 8, color, 1, cv2.LINE_AA)
    cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
    cv2.putText(image, label, (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _draw_projected_marker(image: np.ndarray, pixel: list[int] | None, color: tuple, label: str) -> None:
    if pixel is None:
        return
    x, y = int(pixel[0]), int(pixel[1])
    cv2.circle(image, (x, y), 5, color, 1, cv2.LINE_AA)
    cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=1)
    cv2.putText(image, label, (x + 6, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
