"""Shared training and deployment contract; does not import mjlab."""

from pathlib import Path

LEGS = ("FR", "FL", "RL", "RR")
JOINTS = tuple(f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf"))
DEFAULT = (0.0, 0.95, -1.7) * 4
OBS_ORDER = (
    "angular_velocity",
    "projected_gravity",
    "command",
    "joint_position_relative",
    "joint_velocity",
    "last_action",
)
OBS_DIM = 45
HISTORY = 6
KP, KD, TORQUE_LIMIT, ACTION_SCALE = 20.0, 0.5, 27.0, 0.25
PHYSICS_DT, DECIMATION = 0.005, 4
SOURCE_COMMIT = "0ab74ef5d6048345db7a541316019100416b57d6"
ASSET_DIR = Path(__file__).parent / "assets/little_white_v3"
ROBOT_XML = ASSET_DIR / "robot.xml"
