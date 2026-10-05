"""Task identity and checkpoint compatibility for the rough-task revision."""

TASK_VERSION = 2
COMMAND_FRAME = "body_yaw_velocity"


def validate_checkpoint(state, expected_spec=None):
    if state.get("task_version") != TASK_VERSION or state.get("command_frame") != COMMAND_FRAME:
        raise ValueError(
            "This checkpoint belongs to the old directed task; train a new rough policy."
        )
    if expected_spec is not None and state.get("task_spec") != expected_spec:
        raise ValueError(
            "Resume requires identical task, terrain, commands, num-envs, seed and beta."
        )
