"""基于ICP点云匹配的物体位姿估计器。

通过世界坐标系Y条带过滤 + 桌面平面拟合 + cKDTree近邻匹配
实现已知物体的6DoF定位。
"""
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    DETECTION_EXTRACTION_ORDER,
    DETECTION_SPECS,
    GRASP_TEMPLATES,
    MODEL_SEED_ROTATIONS,
    SystemParams,
    GraspTemplate,
    _VENDOR_DIR,
)


@dataclass
class MeshModel:
    """单物体的模板点云模型。"""
    name: str
    points_local: np.ndarray
    colors_local: np.ndarray | None
    seed_rotation: np.ndarray
    template: GraspTemplate


class MeshPoseEstimator:
    """基于模板匹配的物体位姿估计器。"""

    def __init__(self, params: SystemParams, fx: float):
        self.params = params
        self.fx = fx
        self._models = self._load_mesh_models()

    def extract_candidates(
        self,
        rgb: np.ndarray,
        points: dict,
        workspace_mask: np.ndarray,
        compute_grasp_anchor,
    ) -> tuple[list[dict], dict]:
        """从RGB-D数据中提取各物体的候选抓取信息。"""
        base_points = points["points_base"]
        table_mask, strip_masks, stats = self._build_world_object_masks(rgb, points, workspace_mask)
        debug = {
            "table_mask": table_mask.astype(np.uint8),
            "strip_masks": {name: mask.astype(np.uint8) for name, mask in strip_masks.items()},
            "bands": {},
            "base_stats": stats,
            "component_masks": {},
        }
        if int(np.count_nonzero(table_mask)) < 120:
            return [], debug

        candidates: list[dict] = []
        debug["component_stats"] = {}
        for object_type in DETECTION_EXTRACTION_ORDER:
            strip_mask = strip_masks.get(object_type)
            if strip_mask is None or int(np.count_nonzero(strip_mask)) < 120:
                continue
            labeled, num_labels = ndimage.label(strip_mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
            component_summaries: list[dict] = []
            component_masks: list[dict] = []
            best_component = None
            for label_idx in range(1, num_labels + 1):
                component_mask = labeled == label_idx
                component_size = int(np.count_nonzero(component_mask))
                if component_size < 120:
                    continue
                component_masks.append(
                    {
                        "label": int(label_idx),
                        "mask": component_mask.astype(np.uint8),
                    }
                )
                component_points_base = base_points[component_mask]
                component_colors = rgb[component_mask].astype(np.float64) / 255.0
                pose, _ = self._match_cluster_to_model(
                    component_points_base,
                    cluster_colors=component_colors,
                    allowed_model_names={object_type},
                )
                summary = {
                    "label": int(label_idx),
                    "component_pixels": component_size,
                    "matched": False,
                }
                if pose is None:
                    component_summaries.append(summary)
                    continue
                model_pts = pose["points_world"]
                tree_local = cKDTree(model_pts)
                comp_dist, _ = tree_local.query(component_points_base, k=1, workers=-1)
                local_keep = comp_dist < float(self.params.mesh_mask_distance_m)
                kept_count = int(np.count_nonzero(local_keep))
                coverage = float(kept_count) / float(max(1, component_size))
                size_bonus = 0.03 * float(np.log1p(kept_count))
                coverage_bonus = 0.10 * coverage
                selection_score = float(pose["score"]) + size_bonus + coverage_bonus
                summary.update(
                    {
                        "matched": True,
                        "kept_pixels": kept_count,
                        "score": float(pose["score"]),
                        "selection_score": selection_score,
                        "fitness": float(pose["fitness"]),
                        "rmse": float(pose["rmse"]),
                        "coverage": coverage,
                        "size_bonus": size_bonus,
                        "coverage_bonus": coverage_bonus,
                    }
                )
                component_summaries.append(summary)
                if kept_count < 120:
                    continue
                if best_component is None or selection_score > float(best_component["selection_score"]):
                    best_component = {
                        "pose": pose,
                        "component_mask": component_mask,
                        "local_keep": local_keep,
                        "label": int(label_idx),
                        "selection_score": selection_score,
                    }

            debug["component_stats"][object_type] = component_summaries
            debug["component_masks"][object_type] = component_masks
            if best_component is None:
                continue

            best_pose = best_component["pose"]
            comp_mask = np.zeros_like(table_mask, dtype=bool)
            component_pixels = np.argwhere(best_component["component_mask"])
            kept_pixels = component_pixels[best_component["local_keep"]]
            comp_mask[kept_pixels[:, 0], kept_pixels[:, 1]] = True
            debug["bands"][object_type] = comp_mask.astype(np.uint8)
            debug.setdefault("selected_component_label", {})[object_type] = int(best_component["label"])

            ys, xs = np.where(comp_mask)
            py = int(np.clip(round(float(np.median(ys))), 0, IMAGE_HEIGHT - 1))
            px = int(np.clip(round(float(np.median(xs))), 0, IMAGE_WIDTH - 1))
            depth_m = float(np.median(points["points_cam_ros"][comp_mask, 2]))

            top_idx = int(np.argmax(model_pts[:, 2]))
            top_point_base = np.asarray(model_pts[top_idx], dtype=np.float32)
            pose_cfg = DETECTION_SPECS[best_pose["object_type"]]
            object_center_base = np.asarray(best_pose["object_center_base"], dtype=np.float32).copy()
            object_center_base[:2] += np.asarray(pose_cfg.fixed_center_xy_offset_base_m, dtype=np.float32)
            object_center_base[2] = float(pose_cfg.fixed_center_z_base_m)
            sugar_x_fit_debug = None
            sugar_front_face_x = None
            if best_pose["object_type"] == "sugar":
                sugar_center_x, sugar_x_fit_debug = self._estimate_sugar_center_x(
                    component_points_base=base_points[best_component["component_mask"]],
                    rotated_model_pts=best_pose["points_world"] - best_pose["translation"],
                )
                if np.isfinite(sugar_center_x):
                    object_center_base[0] = np.float32(sugar_center_x)
                    sugar_front_face_x = float(sugar_x_fit_debug.get("front_face_x_base")) if sugar_x_fit_debug is not None else None
            yaw = float(math.atan2(best_pose["rotation"][1, 0], best_pose["rotation"][0, 0]))
            candidates.append(
                {
                    "pixel": [px, py],
                    "quality": float(best_pose["score"]),
                    "angle": yaw,
                    "depth_m": depth_m,
                    "point_cam_ros": np.median(points["points_cam_ros"][comp_mask], axis=0).astype(np.float32),
                    "point_cam_frame": np.median(points["points_cam_frame"][comp_mask], axis=0).astype(np.float32),
                    "surface_base": top_point_base,
                    "object_center_base": object_center_base,
                    "grasp_anchor_base": compute_grasp_anchor(
                        best_pose["object_type"],
                        object_center_base,
                    ),
                    "object_type": best_pose["object_type"],
                    "candidate_source": "mesh_pose",
                    "estimated_pose_yaw_rad": yaw,
                    "mesh_pose_fitness": float(best_pose["fitness"]),
                    "mesh_pose_rmse": float(best_pose["rmse"]),
                    "mesh_pose_translation": np.asarray(best_pose["translation"], dtype=np.float32),
                    "mesh_pose_rotation": np.asarray(best_pose["rotation"], dtype=np.float32),
                    "mesh_object_center_base": object_center_base,
                    "mesh_object_rotation": np.asarray(best_pose["object_rotation"], dtype=np.float32),
                    "mesh_top_point_base": top_point_base,
                    "sugar_front_face_x_base": sugar_front_face_x if best_pose["object_type"] == "sugar" else None,
                    "sugar_x_fit_debug": sugar_x_fit_debug,
                    "cluster_centroid_base": np.asarray(best_pose["cluster_centroid"], dtype=np.float32),
                    "icp_model_center_base": np.median(model_pts, axis=0).astype(np.float32),
                }
            )

        order = {name: idx for idx, name in enumerate(DETECTION_EXTRACTION_ORDER)}
        candidates.sort(key=lambda c: (order.get(c["object_type"], 99), -c["quality"]))
        return candidates, debug

    def _load_mesh_models(self) -> dict[str, MeshModel]:
        """加载各物体的模板点云。"""
        models: dict[str, MeshModel] = {}
        for name, template in GRASP_TEMPLATES.items():
            pose_cfg = DETECTION_SPECS[name]
            model_points_path = pose_cfg.model_points_path
            if not os.path.isfile(model_points_path):
                model_points_path = os.path.join(_VENDOR_DIR, "object_model_points", f"{name}.npy")
            points_local = np.load(model_points_path).astype(np.float64)
            points_local, colors_local = self._downsample_with_colors(points_local, None, self.params.mesh_model_max_points)
            models[name] = MeshModel(
                name=name,
                points_local=points_local,
                colors_local=colors_local,
                seed_rotation=MODEL_SEED_ROTATIONS[name],
                template=template,
            )
        return models

    def _downsample_with_colors(
        self,
        points: np.ndarray,
        colors: np.ndarray | None,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """均匀降采样点云及对应颜色。"""
        if points.shape[0] <= max_points:
            return points, colors
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
        pts = points[idx]
        cols = colors[idx] if colors is not None else None
        return pts, cols

    def _match_cluster_to_model(
        self,
        cluster_pts: np.ndarray,
        cluster_colors: np.ndarray | None = None,
        allowed_model_names: set[str] | None = None,
    ) -> tuple[dict | None, np.ndarray | None]:
        """将点云簇与模板进行匹配，返回最佳匹配结果。"""
        if cluster_pts.shape[0] < 120:
            return None, None
        cluster_pts = np.asarray(cluster_pts, dtype=np.float64)
        if cluster_colors is not None:
            cluster_colors = np.clip(np.asarray(cluster_colors, dtype=np.float64), 0.0, 1.0)
            if cluster_colors.shape[0] != cluster_pts.shape[0]:
                cluster_colors = None
        cluster_pts, cluster_colors = self._downsample_with_colors(
            cluster_pts,
            cluster_colors,
            self.params.mesh_scene_max_points,
        )
        cluster_centroid = np.median(cluster_pts, axis=0)

        best = None
        best_cloud = None
        for model in self._models.values():
            if allowed_model_names is not None and model.name not in allowed_model_names:
                continue
            rotated_model_pts = model.points_local @ model.seed_rotation.T
            if model.name == "sugar":
                translation = self._fit_box_translation(cluster_pts, rotated_model_pts)
                tpts = rotated_model_pts + translation
                pose_rotation = np.eye(3, dtype=np.float64)
                object_rotation = model.seed_rotation.copy()
            else:
                model_median = np.median(rotated_model_pts, axis=0)
                translation = cluster_centroid - model_median
                tpts = rotated_model_pts + translation
                pose_rotation = np.eye(3, dtype=np.float64)
                object_rotation = model.seed_rotation.copy()
            if tpts.size == 0:
                continue
            tree_model = cKDTree(tpts)
            target_to_model_dist, _ = tree_model.query(cluster_pts, k=1, workers=-1)
            threshold = float(self.params.mesh_icp_threshold_m)
            inlier_mask = target_to_model_dist < threshold
            fitness = float(np.count_nonzero(inlier_mask)) / float(max(1, target_to_model_dist.shape[0]))
            if np.any(inlier_mask):
                rmse = float(np.sqrt(np.mean(np.square(target_to_model_dist[inlier_mask]))))
            else:
                rmse = float("inf")
            score = fitness - rmse * 4.0 if np.isfinite(rmse) else -1e9
            pose = {
                "object_type": model.name,
                "fitness": fitness,
                "rmse": rmse,
                "score": score,
                "rotation": pose_rotation,
                "translation": translation.copy(),
                "object_rotation": object_rotation.copy(),
                "object_center_base": translation.copy(),
                "cluster_centroid": cluster_centroid.copy(),
                "points_world": tpts,
                "template": model.template,
                "fit_method": "box_fit" if model.name == "sugar" else "median_shift",
            }
            if best is None or pose["score"] > best["score"]:
                best = pose
                best_cloud = tpts
        return best, best_cloud

    def _fit_box_translation(self, cluster_pts: np.ndarray, rotated_model_pts: np.ndarray) -> np.ndarray:
        """对糖盒使用包围盒鲁棒对齐。"""
        model_min = np.min(rotated_model_pts, axis=0)
        model_max = np.max(rotated_model_pts, axis=0)
        model_extent = model_max - model_min
        model_center = 0.5 * (model_min + model_max)

        center = np.zeros(3, dtype=np.float64)
        quantiles = np.percentile(cluster_pts, [2.0, 50.0, 98.0], axis=0)
        obs_lo = quantiles[0]
        obs_med = quantiles[1]
        obs_hi = quantiles[2]
        obs_extent = obs_hi - obs_lo

        for axis in range(3):
            expected_extent = float(model_extent[axis])
            if expected_extent <= 1e-6:
                center[axis] = obs_med[axis]
                continue
            if obs_extent[axis] < 0.88 * expected_extent:
                left_span = float(obs_med[axis] - obs_lo[axis])
                right_span = float(obs_hi[axis] - obs_med[axis])
                if left_span >= right_span:
                    center[axis] = float(obs_lo[axis] + 0.5 * expected_extent)
                else:
                    center[axis] = float(obs_hi[axis] - 0.5 * expected_extent)
            else:
                center[axis] = float(0.5 * (obs_lo[axis] + obs_hi[axis]))

        return center - model_center

    def _estimate_sugar_center_x(
        self,
        *,
        component_points_base: np.ndarray,
        rotated_model_pts: np.ndarray,
    ) -> tuple[float, dict | None]:
        """通过可见面估算糖盒的X方向中心。"""
        component_points_base = np.asarray(component_points_base, dtype=np.float64)
        rotated_model_pts = np.asarray(rotated_model_pts, dtype=np.float64)
        if component_points_base.shape[0] < 32 or rotated_model_pts.shape[0] < 8:
            return float("nan"), None

        model_x_extent = float(np.max(rotated_model_pts[:, 0]) - np.min(rotated_model_pts[:, 0]))
        model_z_extent = float(np.max(rotated_model_pts[:, 2]) - np.min(rotated_model_pts[:, 2]))
        if not np.isfinite(model_x_extent) or model_x_extent <= 1e-6:
            return float("nan"), None

        x_hi = float(np.percentile(component_points_base[:, 0], 98.0))
        z_hi = float(np.percentile(component_points_base[:, 2], 98.0))
        front_band = max(0.008, min(0.020, 0.35 * model_x_extent))
        top_band = max(0.006, min(0.018, 0.35 * model_z_extent))
        front_mask = component_points_base[:, 0] >= (x_hi - front_band)
        top_mask = component_points_base[:, 2] >= (z_hi - top_band)
        front_count = int(np.count_nonzero(front_mask))
        top_count = int(np.count_nonzero(top_mask))

        debug = {
            "method": "front_face",
            "front_face_point_count": front_count,
            "top_plane_point_count": top_count,
            "front_band_m": float(front_band),
            "top_band_m": float(top_band),
        }

        front_face_x = float(np.percentile(component_points_base[:, 0], 95.0))
        center_x = front_face_x - 0.5 * model_x_extent
        debug["front_face_x_base"] = front_face_x

        if top_count > front_count and top_count >= 16:
            top_points = component_points_base[top_mask]
            top_x_lo = float(np.percentile(top_points[:, 0], 5.0))
            top_x_hi = float(np.percentile(top_points[:, 0], 95.0))
            center_x = 0.5 * (top_x_lo + top_x_hi)
            debug.update(
                {
                    "method": "top_plane",
                    "top_plane_x_lo_base": top_x_lo,
                    "top_plane_x_hi_base": top_x_hi,
                }
            )

        return center_x, debug

    def _estimate_table_height(
        self,
        z_values: np.ndarray,
        *,
        bin_size_m: float = 0.002,
        low_quantile: float = 0.35,
        min_keep_ratio: float = 0.20,
    ) -> dict:
        """从低处点云估算桌面高度。"""
        z_values = np.asarray(z_values, dtype=np.float64)
        z_values = z_values[np.isfinite(z_values)]
        if z_values.size == 0:
            return {
                "table_cluster_found": False,
                "estimated_table_z_max_m": None,
                "cluster_z_min_m": None,
                "cluster_z_max_m": None,
                "peak_bin_count": 0,
                "num_low_points_considered": 0,
                "bin_size_m": bin_size_m,
            }

        low_cut = float(np.quantile(z_values, low_quantile))
        low_points = z_values[z_values <= low_cut]
        if low_points.size == 0:
            return {
                "table_cluster_found": False,
                "estimated_table_z_max_m": None,
                "cluster_z_min_m": None,
                "cluster_z_max_m": None,
                "peak_bin_count": 0,
                "num_low_points_considered": 0,
                "bin_size_m": bin_size_m,
            }

        z_min = float(np.min(low_points))
        z_max = float(np.max(low_points))
        num_bins = max(4, int(np.ceil((z_max - z_min) / bin_size_m)) + 1)
        hist, edges = np.histogram(low_points, bins=num_bins, range=(z_min, z_max + 1e-9))
        peak_idx = int(np.argmax(hist))
        peak_count = int(hist[peak_idx])
        if peak_count <= 0:
            return {
                "table_cluster_found": False,
                "estimated_table_z_max_m": None,
                "cluster_z_min_m": None,
                "cluster_z_max_m": None,
                "peak_bin_count": 0,
                "num_low_points_considered": int(low_points.size),
                "bin_size_m": bin_size_m,
            }

        keep_threshold = max(1, int(np.ceil(peak_count * min_keep_ratio)))
        left = peak_idx
        right = peak_idx
        while left - 1 >= 0 and hist[left - 1] >= keep_threshold:
            left -= 1
        while right + 1 < hist.shape[0] and hist[right + 1] >= keep_threshold:
            right += 1

        cluster_lo = float(edges[left])
        cluster_hi = float(edges[right + 1])
        return {
            "table_cluster_found": True,
            "estimated_table_z_max_m": cluster_hi,
            "cluster_z_min_m": cluster_lo,
            "cluster_z_max_m": cluster_hi,
            "peak_bin_count": peak_count,
            "keep_threshold": keep_threshold,
            "num_low_points_considered": int(low_points.size),
            "bin_size_m": bin_size_m,
            "low_quantile": low_quantile,
        }

    def _fit_table_plane(self, points_xyz: np.ndarray) -> dict:
        """最小二乘拟合桌面平面 z = ax + by + c。"""
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        valid = np.all(np.isfinite(points_xyz), axis=1)
        points_xyz = points_xyz[valid]
        if points_xyz.shape[0] < 16:
            return {
                "plane_found": False,
                "coeff_abc": None,
                "normal_xyz": None,
                "num_plane_points": int(points_xyz.shape[0]),
                "rmse_m": None,
            }

        A = np.column_stack([points_xyz[:, 0], points_xyz[:, 1], np.ones(points_xyz.shape[0], dtype=np.float64)])
        b = points_xyz[:, 2]
        coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, by, c = coeff.tolist()
        pred = A @ coeff
        residual = b - pred
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        normal = np.array([-a, -by, 1.0], dtype=np.float64)
        normal /= max(np.linalg.norm(normal), 1e-12)
        return {
            "plane_found": True,
            "coeff_abc": [float(a), float(by), float(c)],
            "normal_xyz": normal.tolist(),
            "num_plane_points": int(points_xyz.shape[0]),
            "rmse_m": rmse,
        }

    def _build_world_object_masks(
        self,
        rgb: np.ndarray,
        points: dict,
        workspace_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
        """构建世界坐标系下的物体检测掩码。"""
        world_points = points["points_world"]
        finite_mask = np.all(np.isfinite(world_points), axis=-1)
        x_mask = (
            np.isfinite(world_points[..., 0])
            & (world_points[..., 0] >= float(self.params.object_world_x_min_m))
            & (world_points[..., 0] <= float(self.params.object_world_x_max_m))
        )
        pre_z_mask = (
            np.isfinite(world_points[..., 2])
            & (world_points[..., 2] <= float(self.params.object_world_z_max_m))
        )
        banana_y_min, _banana_y_max = DETECTION_SPECS["banana"].world_strip_y_range
        table_seed_y_max = float(banana_y_min - 0.02)
        table_seed_y_min = float(table_seed_y_max - 0.20)
        table_seed_y_mask = (
            np.isfinite(world_points[..., 1])
            & (world_points[..., 1] >= table_seed_y_min)
            & (world_points[..., 1] <= table_seed_y_max)
        )
        low_pool_mask = workspace_mask & finite_mask & x_mask & pre_z_mask & table_seed_y_mask
        low_pool_z = world_points[..., 2][low_pool_mask]
        table_stats = self._estimate_table_height(low_pool_z)
        if table_stats["table_cluster_found"]:
            cluster_lo = float(table_stats["cluster_z_min_m"])
            cluster_hi = float(table_stats["cluster_z_max_m"])
            plane_seed_mask = low_pool_mask & (world_points[..., 2] >= cluster_lo) & (world_points[..., 2] <= cluster_hi)
        else:
            plane_seed_mask = low_pool_mask & (world_points[..., 2] >= float(self.params.object_world_z_min_m)) & (
                world_points[..., 2] <= float(self.params.object_world_z_min_m) + 0.03
            )

        plane_seed_points = world_points[plane_seed_mask]
        plane_stats = self._fit_table_plane(plane_seed_points)
        table_clearance_m = 0.006
        if plane_stats["plane_found"]:
            a, b, c = plane_stats["coeff_abc"]
            plane_z = a * world_points[..., 0] + b * world_points[..., 1] + c
            z_mask = (
                np.isfinite(world_points[..., 2])
                & np.isfinite(plane_z)
                & ((world_points[..., 2] - plane_z) > table_clearance_m)
                & (world_points[..., 2] <= float(self.params.object_world_z_max_m))
            )
        else:
            estimated_table_z_max_m = float(table_stats["estimated_table_z_max_m"]) if table_stats["table_cluster_found"] else float(
                self.params.object_world_z_min_m
            )
            z_mask = (
                np.isfinite(world_points[..., 2])
                & (world_points[..., 2] > (estimated_table_z_max_m + table_clearance_m))
                & (world_points[..., 2] <= float(self.params.object_world_z_max_m))
            )

        common_mask = workspace_mask & finite_mask & x_mask & z_mask
        table_mask = common_mask

        strip_masks: dict[str, np.ndarray] = {}
        y_pad = float(self.params.object_world_strip_pad_m)
        one_sided_y_extra = {
            "sugar": (0.0, 0.07),
            "mustard": (0.0, 0.0),
            "banana": (0.04, 0.0),
        }
        for name, pose_cfg in DETECTION_SPECS.items():
            y_min, y_max = pose_cfg.world_strip_y_range
            extra_lo, extra_hi = one_sided_y_extra.get(name, (0.0, 0.0))
            y_mask = (
                np.isfinite(world_points[..., 1])
                & (world_points[..., 1] >= (y_min - y_pad - extra_lo))
                & (world_points[..., 1] <= (y_max + y_pad + extra_hi))
            )
            strip_masks[name] = common_mask & y_mask

        stats = {}
        all_valid_points = world_points[workspace_mask & finite_mask]
        table_points = world_points[table_mask]
        if all_valid_points.size:
            stats["world_valid_bounds"] = {
                "x_min": float(np.min(all_valid_points[:, 0])),
                "x_max": float(np.max(all_valid_points[:, 0])),
                "y_min": float(np.min(all_valid_points[:, 1])),
                "y_max": float(np.max(all_valid_points[:, 1])),
                "z_min": float(np.min(all_valid_points[:, 2])),
                "z_max": float(np.max(all_valid_points[:, 2])),
                "x_med": float(np.median(all_valid_points[:, 0])),
                "y_med": float(np.median(all_valid_points[:, 1])),
                "z_med": float(np.median(all_valid_points[:, 2])),
            }
        stats["mask_counts"] = {
            "workspace": int(np.count_nonzero(workspace_mask)),
            "finite": int(np.count_nonzero(finite_mask)),
            "x_range": int(np.count_nonzero(x_mask)),
            "pre_z_range": int(np.count_nonzero(pre_z_mask)),
            "table_seed_y": int(np.count_nonzero(table_seed_y_mask)),
            "plane_seed": int(np.count_nonzero(plane_seed_mask)),
            "z_range": int(np.count_nonzero(z_mask)),
            "low_pool": int(np.count_nonzero(low_pool_mask)),
            "workspace_finite": int(np.count_nonzero(workspace_mask & finite_mask)),
            "table": int(np.count_nonzero(table_mask)),
        }
        stats["table_height_estimation"] = table_stats
        stats["table_plane_fit"] = plane_stats
        stats["table_seed_y_range_m"] = {
            "y_min": float(table_seed_y_min),
            "y_max": float(table_seed_y_max),
        }
        stats["z_thresholds"] = {
            "configured_object_world_z_min_m": float(self.params.object_world_z_min_m),
            "configured_object_world_z_max_m": float(self.params.object_world_z_max_m),
            "table_clearance_m": float(table_clearance_m),
        }
        stats["one_sided_y_extra_m"] = {
            name: {"extra_lo": float(extra[0]), "extra_hi": float(extra[1])}
            for name, extra in one_sided_y_extra.items()
        }
        if table_points.size:
            stats["table_bounds"] = {
                "x_min": float(np.min(table_points[:, 0])),
                "x_max": float(np.max(table_points[:, 0])),
                "y_min": float(np.min(table_points[:, 1])),
                "y_max": float(np.max(table_points[:, 1])),
                "z_min": float(np.min(table_points[:, 2])),
                "z_max": float(np.max(table_points[:, 2])),
                "x_med": float(np.median(table_points[:, 0])),
                "y_med": float(np.median(table_points[:, 1])),
                "z_med": float(np.median(table_points[:, 2])),
            }
        return table_mask, strip_masks, stats
