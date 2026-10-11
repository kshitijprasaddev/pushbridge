import numpy as np

from pushbridge.precision_rl import StrictCompactGym, strict_hold_succeeded


def test_strict_training_hold_rejects_transient_escape_even_if_endpoint_returns():
    assert strict_hold_succeeded([.007] * 11)
    assert strict_hold_succeeded([.008] * 11)
    assert not strict_hold_succeeded([.007, .0081] + [.007] * 9)
    assert not strict_hold_succeeded([.007] * 10)
    assert not strict_hold_succeeded([.007] * 10 + [float('nan')])


class ScriptedPhysics:
    def __init__(self, states):
        self.states = iter(states)
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return np.array(next(self.states))


def scripted_env(distances):
    env = StrictCompactGym.__new__(StrictCompactGym)
    env.goal = np.array([.23, .105])
    env.state = np.array([.19, .105, .22, .105])
    states = [np.r_[env.state[:2], env.goal + [distance, 0]] for distance in distances]
    env.physics = ScriptedPhysics(states)
    env.count, env.horizon = 0, 120
    return env


def test_strict_training_uses_zero_action_hold_and_does_not_accept_18mm_endpoint():
    env = scripted_env([.017])
    _, _, terminated, _, info = env.step([0, 0])
    assert not terminated and not info['is_success']
    assert info['hold_control_steps'] == 0
    env = scripted_env([.007] * 11)
    _, _, terminated, _, info = env.step([.3, -.1])
    assert terminated and info['is_success']
    assert info['hold_control_steps'] == 10
    np.testing.assert_array_equal(env.physics.actions[1:], np.zeros((10, 2)))
    env = scripted_env([.007, .0081] + [.007] * 9)
    _, _, terminated, _, info = env.step([0, 0])
    assert not terminated and not info['is_success']
