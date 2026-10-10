import mujoco
import numpy as np
import pytest

from pushbridge.data import load_episodes, write_json
from pushbridge.embodiment import COMPACT_SPEC, CompactPushEnv
from pushbridge.sim import ORIGIN


def test_compact_kinematics_match_physical_model_and_jacobian():
    env = CompactPushEnv()
    rng = np.random.default_rng(91)
    try:
        for _ in range(12):
            q = rng.uniform(env.limits[:, 0]+.1, env.limits[:, 1]-.1)
            env.data.qpos[:3] = q
            mujoco.mj_forward(env.model, env.data)
            position, jacobian = env.fk_jacobian(q)
            np.testing.assert_allclose(position, env.data.site_xpos[env.tip_id, :2], atol=1e-12)
            jacp, jacr = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
            mujoco.mj_jacSite(env.model, env.data, jacp, jacr, env.tip_id)
            np.testing.assert_allclose(jacobian, jacp[:2, :3], atol=1e-12)
    finally:
        env.close()


def test_compact_command_bounds_and_speed_are_enforced():
    env = CompactPushEnv()
    try:
        env.reset()
        rng = np.random.default_rng(53)
        for _ in range(70):
            old = env.command.copy()
            state = env.step(rng.normal(0, .02, 2))
            assert np.isfinite(state).all()
            assert np.all(env.command >= env.limits[:, 0])
            assert np.all(env.command <= env.limits[:, 1])
            assert np.max(np.abs(env.command-old)) <= COMPACT_SPEC['max_command_joint_speed_rad_s']*.1+1e-12
        assert env.adapter_diagnostics['slew_limited_steps'] > 0
    finally:
        env.close()


def test_loader_retains_original_video_timestamps_separately_from_control_time(tmp_path):
    source_time = np.array([2.6003, 2.7002, 2.8004])
    np.savez_compressed(tmp_path/'clip.npz', state=np.zeros((3, 4)), time=np.arange(3)*.1,
                        valid=np.ones(3, bool), video_time_s=source_time)
    write_json(tmp_path/'manifest.json', {'source': 'personal', 'episodes': [{'id': 'clip', 'file': 'clip.npz'}]})
    _, episodes = load_episodes(tmp_path)
    np.testing.assert_array_equal(episodes[0]['video_time_s'], source_time)
    np.testing.assert_allclose(episodes[0]['time'], [0, .1, .2])


def test_exported_torch_actor_matches_sb3_without_loading_sb3_checkpoint(tmp_path):
    pytest.importorskip('stable_baselines3')
    from stable_baselines3 import SAC
    from pushbridge.fast_policy import DirectTorchActor, FastActor, export_actor
    from pushbridge.rl import PushGym, observation
    from pushbridge.sim import MAX_ACTION, clip_action
    env = PushGym()
    try:
        agent = SAC('MlpPolicy', env, policy_kwargs={'net_arch': [64, 64]}, seed=16, device='cpu')
        path = tmp_path/'actor.npz'
        export_actor(agent, path)
        direct, fast = DirectTorchActor(path), FastActor(path)
        rng = np.random.default_rng(33)
        for _ in range(60):
            state = rng.uniform([.01, .01, .02, .02], [.28, .19, .27, .18])
            goal = rng.uniform([.08, .02], [.27, .18])
            output, _ = agent.predict(observation(state, goal), deterministic=True)
            expected = clip_action(output*MAX_ACTION)
            np.testing.assert_array_equal(direct.act(state, goal), expected)
            np.testing.assert_allclose(fast.act(state, goal), expected, atol=4e-9, rtol=0)
    finally:
        env.close()
