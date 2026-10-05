from types import SimpleNamespace

import pytest
import torch

from dreamwaq_mjlab.checkpoint import COMMAND_FRAME, TASK_VERSION, validate_checkpoint
from dreamwaq_mjlab.deployment.evaluate import set_fixed_commands, tracking_score
from dreamwaq_mjlab.deployment.play import HistoryPolicy
from dreamwaq_mjlab.tasks.config import make_cfg, task_spec
from dreamwaq_mjlab.tasks.mdp import tracking_errors
from dreamwaq_mjlab.tasks.terrains import KINDS, terrain_cfg


def test_task_has_full_motion_distribution_and_no_course_terminations():
    cfg = make_cfg(64)
    command = cfg.commands["twist"]
    assert command.heading_command and command.rel_world_envs == 0
    assert command.ranges.lin_vel_x == (-0.6, 1.0)
    assert command.ranges.lin_vel_y == (-0.5, 0.5)
    assert command.resampling_time_range == (10, 10) and command.rel_standing_envs == 0.2
    assert set(cfg.terminations) == {
        "time_out",
        "out_of_terrain_bounds",
        "fell_over",
        "illegal_contact",
    }
    assert cfg.terminations["out_of_terrain_bounds"].time_out
    assert cfg.terminations["illegal_contact"].params["sensor_name"] == "trunk_ground_touch"
    assert "progress" not in cfg.rewards and "thigh_collision" in cfg.rewards
    assert cfg.scene.terrain.max_init_terrain_level == 5
    assert "terrain" in cfg.curriculum
    fixed = make_cfg(8, curriculum=False, difficulty=0.5)
    assert not fixed.curriculum
    assert fixed.scene.terrain.terrain_generator.num_rows == 1
    assert not fixed.observations["critic"].enable_corruption


def test_terrain_weights_and_difficulty_validation():
    cfg = terrain_cfg()
    assert tuple(cfg.sub_terrains) == KINDS
    assert sum(c.proportion for c in cfg.sub_terrains.values()) == pytest.approx(1.0)
    assert all(c.size == (8.0, 8.0) for c in cfg.sub_terrains.values())
    assert cfg.border_width == 20 and cfg.border_height > 0
    with pytest.raises(ValueError):
        terrain_cfg(difficulty=1.1)
    with pytest.raises(ValueError):
        terrain_cfg(kind="stairs_up")


def test_checkpoint_rejects_old_or_different_tasks():
    spec = task_spec(make_cfg(64), "mixed", 42, 1.0)
    state = {"task_version": TASK_VERSION, "command_frame": COMMAND_FRAME, "task_spec": spec}
    validate_checkpoint(state, spec)
    with pytest.raises(ValueError, match="old directed"):
        validate_checkpoint({"format_version": 1})
    with pytest.raises(ValueError, match="identical"):
        validate_checkpoint(state, task_spec(make_cfg(32), "mixed", 42, 1.0))


def test_tracking_uses_yaw_frame_and_world_yaw_rate():
    data = SimpleNamespace(
        heading_w=torch.tensor([torch.pi / 2]),
        root_link_lin_vel_w=torch.tensor([[0.2, 0.4, 0.3]]),
        root_link_ang_vel_w=torch.tensor([[0.4, 0.2, -0.8]]),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    xy, yaw = tracking_errors(env, torch.tensor([[0.4, -0.2, -0.8]]))
    torch.testing.assert_close(xy, torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(yaw, torch.zeros(1))


def test_scoring_penalizes_early_failure_and_not_following_commands():
    perfect = tracking_score(torch.zeros(10), torch.zeros(10))
    assert perfect.mean() == 1
    # The fixed horizon denominator makes a 1-step perfect fall score 1/10.
    assert perfect[:1].sum() / 10 == pytest.approx(0.1)
    stopped = tracking_score(torch.tensor([0.6**2]), torch.tensor([0.8**2]))
    assert stopped.item() < 0.2


def test_fixed_commands_survive_heading_and_standing_modes():
    cmd = SimpleNamespace(
        is_heading_env=torch.ones(3, dtype=torch.bool),
        is_standing_env=torch.ones(3, dtype=torch.bool),
        is_world_env=torch.ones(3, dtype=torch.bool),
        is_forward_env=torch.ones(3, dtype=torch.bool),
        vel_command_b=torch.ones(3, 3),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd))
    set_fixed_commands(env, torch.tensor([0, 2]), torch.tensor([[-0.4, 0, 0], [0, 0, 0.8]]))
    assert cmd.is_heading_env.tolist() == [False, True, False]
    assert cmd.is_standing_env.tolist() == [False, True, False]
    assert cmd.vel_command_b[0, 0] == pytest.approx(-0.4)
    assert cmd.vel_command_b[2, 2] == pytest.approx(0.8)


def test_viewer_history_resets_only_done_world():
    captured = []
    env = SimpleNamespace(episode_length_buf=torch.zeros(2, dtype=torch.long))
    adapter = HistoryPolicy(env, lambda h: captured.append(h.clone()) or torch.zeros(2, 12))
    obs = torch.arange(90).reshape(2, 45).float()
    adapter({"actor": obs})
    env.episode_length_buf[:] = 1
    adapter({"actor": obs + 1})
    env.episode_length_buf[:] = torch.tensor([0, 2])
    adapter({"actor": obs + 2})
    torch.testing.assert_close(captured[-1][0], (obs[0] + 2).repeat(6, 1))
    torch.testing.assert_close(captured[-1][1, -2], obs[1] + 1)


def test_distance_curriculum_all_directions_and_first_reset():
    from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel

    updates = []
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=terrain_cfg()),
        terrain_levels=torch.full((5,), 2),
        terrain_types=torch.arange(5),
        terrain_origins=torch.zeros(10, 9, 3),
        update_env_origins=lambda ids, up, down: updates.append((up.clone(), down.clone())),
    )

    class Scene(dict):
        pass

    scene = Scene(
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor(
                    [[5.0, 0, 0], [0, -5.0, 0], [0, 0, 0], [0, 0, 0], [1.0, 0, 0]]
                )
            )
        )
    )
    scene.terrain = terrain
    scene.env_origins = torch.zeros(5, 3)
    command = torch.tensor([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 0], [0, 0, 0.8], [0.2, 0, 0]])
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_command=lambda _: command),
        max_episode_length_s=20,
        common_step_counter=0,
    )
    terrain_levels_vel(env, torch.arange(5), "twist")
    assert not updates[-1][0].any() and not updates[-1][1].any()
    env.common_step_counter = 1
    terrain_levels_vel(env, torch.arange(5), "twist")
    assert updates[-1][0].tolist() == [True, True, False, False, False]
    assert updates[-1][1].tolist() == [False, False, False, False, True]


def test_command_intervention_refreshes_cached_policy_input_without_losing_noise():
    from dreamwaq_mjlab.deployment.evaluate import policy_observations

    original = torch.zeros(2, 45)
    original[:, 6:9] = torch.tensor([[0.4, 0, 0.2], [0.4, 0, 0.2]])
    noisy = original + 0.01
    cached = {"actor": noisy.clone(), "clean": original.clone()}
    command = torch.tensor([[-0.4, 0, 0], [0, 0, 0.8]])
    env = SimpleNamespace(
        get_observations=lambda: cached,
        command_manager=SimpleNamespace(get_command=lambda _: command),
    )
    obs = policy_observations(env)
    torch.testing.assert_close(obs["actor"][:, 6:9], command + 0.01)
    torch.testing.assert_close(obs["actor"][:, 9:], noisy[:, 9:])
    torch.testing.assert_close(cached["actor"], noisy)


def test_native_viewer_cleanup_joins_its_daemon_thread(monkeypatch):
    from dreamwaq_mjlab.deployment import viewer_runtime

    calls = []

    def target():
        pass

    target.__module__ = "mujoco.viewer"
    target.__name__ = "_launch_internal"
    thread = SimpleNamespace(
        _target=target, join=lambda timeout: calls.append(timeout), is_alive=lambda: False
    )
    monkeypatch.setattr(viewer_runtime.threading, "enumerate", lambda: [thread])
    viewer_runtime.join_native_viewer_threads()
    assert calls == [10.0]
