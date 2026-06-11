# Task E 重构方案 — 桌面物体抓取放置

## 概述

本方案实现 ATEC 2026 仿真赛道二 Task E：使用 Piper 6DOF 机械臂从桌面抓取 3 个物体（mustard、sugar、banana）并放入篮子。

**方法**：基于 Drake IK 求解器的经典控制流水线（非学习方法），开环批量规划 + 逐物体执行。

**核心流程**：
1. 移动到观察位 → 拍摄 RGB-D 图像
2. 基于 ICP 点云匹配估计各物体位姿
3. 一次性规划所有物体的抓取动作序列
4. 逐物体执行：接近 → 下降 → 夹取 → 抬起 → 放置 → 返回

---

## 文件结构

```
my_refactored/
├── entry.py              (19行)   评估框架入口，AlgSolution 适配器
├── controller.py         (133行)  批量抓取控制器，状态机主循环
├── config.py             (317行)  全局参数、物体模板、检测规格
├── arm_solver.py         (191行)  Drake IK 逆运动学求解器
├── camera_transform.py   (77行)   相机内参、深度反投影、像素投影
├── motion_planner.py     (196行)  运动规划编排 + 动作序列构建
├── pose_estimator.py     (630行)  基于 ICP 的物体 6DoF 位姿估计
├── grasp_selector.py     (301行)  抓取候选点评估与 IK 排序
├── visualizer.py         (260行)  调试可视化帧导出
└── README.md                      本文件
```

总计约 2124 行 Python 代码。

---

## 模块说明

### entry.py — 入口适配器

提供 `AlgSolution` 类，实现官方要求的接口：
- `predicts(obs, current_score)` → `{"action": [...], "giveup": False}`
- `get_action_spec()` → `None`（使用默认动作配置）
- `reset(**kwargs)`

### controller.py — BatchGraspController

批量循环控制器，核心状态机逻辑：
- 首次调用时移动到观察位（`OBSERVE_JOINT_POS`）
- 到达观察位后调用 `_plan_all_objects()` 一次性规划 3 个物体
- 缓存规划结果，逐个执行 pick→place 循环
- 每完成一个物体切换到下一个缓存计划

### config.py — SystemParams + 模板配置

包含所有运行参数：
- `SystemParams` 数据类：控制器时序、IK 容差、运动插值步数等
- `GraspTemplate`：每个物体的抓取姿态模板（角度、偏移、预抓取距离）
- `DetectionSpec`：每个物体的 ICP 检测参数（条带范围、种子旋转、偏移修正）
- 场景常量：桌面尺寸、相机参数、关节预设

### arm_solver.py — PiperArmIK

基于 pydrake 的逆运动学求解器：
- 加载 Piper URDF 建立运动学模型
- `solve_waypoint(target_pos, target_rot, seed_q)` → 求解关节角
- `compute_camera_pose(q)` → 正运动学计算相机位姿
- 坐标系转换：仿真坐标系 ↔ Drake 坐标系

### camera_transform.py — 相机工具

从 RGB-D 图像获取 3D 点云：
- `project_depth_to_base(depth, camera_pose)` → base 坐标系下的点云
- `build_workspace_mask(points)` → 工作空间过滤掩码
- `project_base_to_pixel(point, camera_pose)` → 3D 点投影到像素坐标

### motion_planner.py — MotionSequenceBuilder

规划编排器，将感知结果转化为可执行动作：
- `generate_grasp_plan(obs, q, target_object)` → 规划单个物体的抓取
- `build_action_sequence(q, best_candidate)` → 生成完整动作队列
- 动作序列：接近(8步) → 下降(25步) → 闭合(6步) → 抬起(6步) → 放置(10步) → 释放(10步) → 返回(18步)

### pose_estimator.py — MeshPoseEstimator

ICP 点云配准定位物体：
- 加载预采样的物体点云模型（vendor/object_model_points/）
- 世界坐标系 Y 轴条带过滤：按 Y 范围分离不同物体
- 桌面平面拟合 → 去除桌面点 → 提取物体点云
- ICP 迭代最近点 → 估计 6DoF 位姿
- sugar 特殊处理：box fit 平移修正

### grasp_selector.py — CandidateEvaluator

评估抓取候选并排序：
- `rank_candidates(candidates, current_q)` → IK 可达性排序
- 根据物体位置自动选择模板（远/近/中间插值）
- 计算预抓取位姿 + Y 轴 fallback 搜索（sugar 专用）
- 动态 pitch 和 offset 插值

### visualizer.py — 调试可视化

导出调试帧到文件（`debug_mesh_pose/` 目录）：
- RGB 图像 + 物体检测标注
- 点云投影可视化
- 抓取位姿标注

---

## 依赖

```
drake          # IK 求解器
scipy          # cKDTree (ICP 最近邻)
numpy          # 数值计算
opencv-python  # 图像处理（调试可视化）
torch          # 观测数据处理（GPU tensor）
```

运行时还需要 `vendor/` 目录（与本目录同级）：
- `vendor/piper_description/` — Piper 机械臂 URDF
- `vendor/object_model_points/` — 物体点云模型（sugar.npy, mustard.npy, banana.npy）

---

## 运行测试

### 环境准备

```bash
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/ATEC2026_Simulation_Challenge
conda activate env_isaaclab
```

### 官方测评命令（正式提交用）

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$(pwd):$(pwd)/scripts:$PYTHONPATH" \
python scripts/play_atec_task.py \
  --task ATEC-TaskE-Piper \
  --headless \
  --enable_cameras \
  --debug
```

**说明**：
- 使用 `demo/solution.py` 作为入口（确保其中导入指向本模块）
- `--debug` 每步打印分数（可选）
- 默认步数 200，可加 `--video_length 500` 增加
- 满分 18.00（3 个物体 × 6 分）

### 开发测试命令（带视频录制 + 快速调试）

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$(pwd):$(pwd)/scripts:$PYTHONPATH" \
python scripts/record_task_e.py \
  --solution refactored \
  --video_length 500 \
  --debug
```

**说明**：
- `--solution refactored` 直接加载本模块，无需修改 solution.py
- `--solution senior` 可对比学长代码
- 录制视频保存到 `logs/videos/TaskE-refactored/play/`
- 500 步约 10 秒仿真时间

### 快速验证（不录视频）

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$(pwd):$(pwd)/scripts:$PYTHONPATH" \
python scripts/play_atec_task.py \
  --task ATEC-TaskE-Piper \
  --headless \
  --enable_cameras \
  --video_length 500
```

---

## 参数调优说明

相较于原始实现，本方案对以下运动参数做了小幅调整以增加安全裕度：

| 参数 | 原始值 | 本方案 | 调整理由 |
|------|--------|--------|----------|
| `approach_steps` | 7 | 8 | 接近阶段更平滑 |
| `close_steps` | 5 | 6 | 夹爪闭合时物体更稳定 |
| `lift_steps` | 5 | 6 | 抬升更平稳，减少甩飞风险 |

其余参数（descend_steps=25, place_steps=10, release_steps=10, return_steps=18 等）保持不变。

---

## 提交方式

确保 `demo/solution.py` 内容为：

```python
try:
    from .my_refactored.entry import AlgSolution  # noqa: F401
except ImportError:
    from my_refactored.entry import AlgSolution  # noqa: F401
```

提交时将 `demo/` 目录整体打包上传（包含 Dockerfile、server.py、my_refactored/、vendor/ 等）。

---

## 测试结果

| 测试方式 | 分数 | 说明 |
|---------|------|------|
| play_atec_task.py (官方) | 18.00 | 满分 |
| record_task_e.py Run 1 | 18.00 | 满分 |
| record_task_e.py Run 2 | 18.00 | 满分 |
| record_task_e.py Run 3 | 18.00 | 满分 |
