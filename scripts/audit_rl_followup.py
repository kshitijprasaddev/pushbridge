"""Verify the longer-training follow-up without retraining or rendering."""
import argparse
import json
from pathlib import Path

import numpy as np

from pushbridge.data import sha256, write_json
from pushbridge.embodiment import CONTACT_SCALE, CompactPushEnv
from pushbridge.evaluation import make_trials
from pushbridge.precision import FIT_DISTANCE, HOLD_STEPS
from pushbridge.sim import MAX_ACTION, ORIGIN


ROOT = Path(__file__).resolve().parents[1]


def audit(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    evaluation = json.loads((out / 'evaluation.json').read_text())
    assert evaluation['protocol'] == protocol
    for parent in protocol['parent_checkpoints'].values():
        assert sha256(ROOT / parent['path']) == parent['sha256']
    training = json.loads((out / 'training_summary.json').read_text())
    assert len(training['models']) == 6
    assert training['new_training_decisions_total'] == protocol['added_decisions_total'] == 240000
    weights = {}
    for report in training['models']:
        model_name = f'{report["variant"]}_seed{report["seed"]}'
        path = out / model_name / 'sac.zip'
        assert sha256(path) == report['checkpoint_sha256']
        assert report['new_target_training_decisions'] == 40000
        assert report['target_training_decisions_total'] == 60000
        assert report['parent_sha256'] == protocol['parent_checkpoints'][str(report['seed'])]['sha256']
        weights[model_name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    traces, successes, baseline_bit_identical, maximum_fk_mm = 0, 0, 0, 0.
    for suite, suite_data in evaluation['suites'].items():
        trials = make_trials(20, suite_data['seed'])
        starts = {}
        for name, result in suite_data['results'].items():
            assert result['successes'] == sum(row['success'] for row in result['runs'])
            assert result['episodes'] == len(result['runs']) == 20
            for row in result['runs']:
                trial = trials[row['trial']]
                assert row['friction'] == trial['friction']
                np.testing.assert_array_equal(row['goal'], trial['goal'])
                np.testing.assert_array_equal(row['initial_puck'], trial['puck'])
                np.testing.assert_array_equal(row['initial_tool'], trial['tool'])
                path = out / suite / f'{name}_{row["trial"]:03d}.npz'
                with np.load(path, allow_pickle=False) as trace:
                    state, action, joint, phase, goal = [trace[key] for key in ('state', 'action', 'joint', 'phase', 'goal')]
                assert len(state) == len(action) + 1 == len(joint) == len(phase)
                np.testing.assert_array_equal(goal, trial['goal'])
                if row['trial'] not in starts:
                    starts[row['trial']] = state[0]
                np.testing.assert_array_equal(state[0], starts[row['trial']])
                assert np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(joint).all()
                assert np.linalg.norm(action, axis=1).max(initial=0) <= MAX_ACTION / CONTACT_SCALE + 1e-9
                assert row['adapter']['max_command_joint_speed_rad_s'] <= .25 + 1e-12
                assert row['total_control_steps'] == len(action)
                assert abs(np.linalg.norm(state[-1, 2:] - goal) * 1000 - row['final_distance_mm']) < 1e-8
                assert abs(row['final_clearance_mm'] - (FIT_DISTANCE * 1000 - row['final_distance_mm'])) < 1e-8
                actual_hold_ticks = int(np.count_nonzero(phase == 'verify'))
                assert actual_hold_ticks == row['verification_hold_ticks']
                if row['success']:
                    assert actual_hold_ticks >= HOLD_STEPS
                    np.testing.assert_array_equal(action[-HOLD_STEPS:], np.zeros((HOLD_STEPS, 2)))
                    assert all(value == 'verify' for value in phase[-HOLD_STEPS:])
                    assert np.linalg.norm(state[-HOLD_STEPS-1:, 2:] - goal, axis=1).max() <= FIT_DISTANCE + 1e-12
                    if name.endswith('_finish') or name == 'feedback_only':
                        assert np.count_nonzero(phase == 'release') == 4
                    successes += 1
                positions = np.array([CompactPushEnv.fk_jacobian(q)[0] - ORIGIN for q in joint])
                residual = float(np.linalg.norm(positions - state[:, :2], axis=1).max() * 1000)
                assert residual < .2
                maximum_fk_mm = max(maximum_fk_mm, residual)
                if suite == 'paired' and name.startswith('original_20k'):
                    old_name = name.replace('original_20k', 'posttrained')
                    with np.load(ROOT / 'results/precision_finish/paired' / f'{old_name}_{row["trial"]:03d}.npz', allow_pickle=False) as old:
                        for key, value in zip(('state', 'action', 'joint', 'phase', 'goal'), (state, action, joint, phase, goal)):
                            np.testing.assert_array_equal(value, old[key])
                    baseline_bit_identical += 1
                traces += 1
    assert traces == 760 and baseline_bit_identical == 120
    report = {'status': 'pass', 'scope': 'Follow-up trace/checkpoint/budget verification; no training or rendering.',
              'protocol_sha256': sha256(out / 'protocol.json'), 'trace_count': traces,
              'successful_traces': successes, 'original_baseline_traces_bit_identical': baseline_bit_identical,
              'max_joint_forward_kinematics_residual_mm': maximum_fk_mm,
              'new_training_decisions_total': training['new_training_decisions_total'],
              'new_training_control_steps_total': training['new_training_control_steps_total'],
              'checkpoints': weights,
              'checks': ['Frozen parent hashes', 'Six final models with 40k added decisions each and 240k total decision accounting',
                         'Trace/action/state/joint shapes and limits', 'Strict containment during every successful hold',
                         'Exact paired starts, goals and friction match declared trial seeds',
                         'Four explicit release steps for hand-coded finisher successes',
                         'All 120 paired original-baseline traces bit-identical to frozen evidence']}
    write_json(out / 'audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/rl_precision_followup')
    args = parser.parse_args()
    print(json.dumps(audit(args.out), indent=2))


if __name__ == '__main__':
    main()
