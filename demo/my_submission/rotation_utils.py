"""Pure numpy replacements for pydrake rotation math."""
import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Roll-Pitch-Yaw (XYZ extrinsic = ZYX intrinsic) to 3x3 rotation matrix.

    Matches pydrake's RollPitchYaw(r,p,y).ToRotationMatrix().matrix() convention.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def make_z_rotation(angle_rad: float) -> np.ndarray:
    """Create rotation matrix for rotation about Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def compose_rotation(base_rot: np.ndarray, angle_z: float) -> np.ndarray:
    """Compose a base rotation with a Z-axis rotation.

    Replaces base_rotation.multiply(RotationMatrix.MakeZRotation(angle)).
    """
    return base_rot @ make_z_rotation(angle_z)
