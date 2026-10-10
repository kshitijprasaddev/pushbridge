import numpy as np

from pushbridge.evaluation import make_trials
from pushbridge.precision import PrecisionFeedback, clearance, run_precision


def test_containment_uses_the_object_edge_not_just_its_centre():
    goal = np.array([.230, .105])
    for angle in np.linspace(0, 2*np.pi, 19):
        direction = np.array([np.cos(angle), np.sin(angle)])
        inside = np.r_[goal, goal + direction*.007999]
        outside = np.r_[goal, goal + direction*.008001]
        assert clearance(inside, goal) > 0
        assert clearance(outside, goal) < 0
    old_success = np.r_[goal, goal + [.017, 0]]
    assert clearance(old_success, goal) < 0


def test_precision_push_completes_through_physics_and_holds_after_release():
    trial = make_trials(1, 7131)[0]
    row, trace = run_precision(PrecisionFeedback(), trial)
    assert row['success']
    assert np.linalg.norm(trace['state'][-1, 2:] - trace['state'][0, 2:]) > .05
    assert row['verification_hold_ticks'] == 10
    assert np.count_nonzero(trace['phase'] == 'release') == 4
    verify = trace['state'][trace['phase'] == 'verify']
    assert all(clearance(state, trace['goal']) >= 0 for state in verify)
    np.testing.assert_array_equal(trace['action'][-10:], np.zeros((10, 2)))
    assert row['adapter']['max_command_joint_speed_rad_s'] <= .25 + 1e-12
    assert np.linalg.norm(trace['state'][-1, :2] - trace['state'][-1, 2:]) > .020
