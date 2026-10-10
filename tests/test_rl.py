import numpy as np
import pytest
import torch

pytest.importorskip('stable_baselines3')

from pushbridge.data import sha256, write_json
from pushbridge.model import ContactDynamics
from pushbridge.policy import geometric_action
from pushbridge.rl import PushGym, demonstration_arrays
from pushbridge.sim import MAX_ACTION


def test_gym_seeded_reset_and_success_includes_terminal_hold():
    env = PushGym()
    try:
        first, _ = env.reset(seed=42)
        repeated, _ = env.reset(seed=42)
        np.testing.assert_allclose(first, repeated)
        for _ in range(120):
            obs, reward, terminated, truncated, info = env.step(geometric_action(env.state, env.goal)/MAX_ACTION)
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert terminated and info['is_success']
        assert info['hold_control_steps'] == 10
        assert info['goal_distance_mm'] < 18
    finally:
        env.close()


def test_actor_initialization_excludes_validation_and_test_action_labels(tmp_path):
    episodes = []
    for i in range(5):
        x = .07+np.arange(30)*.002 if i < 3 else .20-np.arange(30)*.002
        state = np.c_[x, np.full(30, .105), np.full(30, .10), np.full(30, .105)]
        name = f'{i}.npz'
        np.savez_compressed(tmp_path/name, state=state, time=np.arange(30)*.1, valid=np.ones(30, bool))
        episodes.append({'id': f'ep{i}', 'file': name})
    write_json(tmp_path/'manifest.json', {'source': 'personal', 'episodes': episodes})
    checkpoint = tmp_path/'model.pt'
    torch.save({'models': [ContactDynamics().state_dict()], 'prior_relative': torch.zeros(1, 2),
                'prior_action': torch.zeros(1, 2), 'train_ids': ['ep0', 'ep1', 'ep2'],
                'validation_ids': ['ep3'], 'test_ids': ['ep4'],
                'manifest_sha256': sha256(tmp_path/'manifest.json')}, checkpoint)
    _, actions, info = demonstration_arrays(tmp_path, checkpoint, 0)
    assert info['training_clip_ids'] == ['ep0', 'ep1', 'ep2']
    assert info['base_labels'] == 87
    # Held-out labels point left. Training labels rotated by at most 60 degrees
    # still point right, so any leaked held-out sample would fail this check.
    assert np.all(actions[:, 0] > 0)
    write_json(tmp_path/'manifest.json', {'source': 'personal', 'episodes': episodes, 'changed': True})
    with pytest.raises(ValueError, match='match'):
        demonstration_arrays(tmp_path, checkpoint, 0)
