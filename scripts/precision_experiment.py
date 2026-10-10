"""Compare unchanged policies with an explicit precision handoff.

Run from the repository root. Uses validation seed 7131, freezes the protocol
before evaluation, and keeps all earlier 18 mm experiments unchanged.
"""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from pushbridge.data import sha256, write_json
from pushbridge.embodiment import ContactAdapter
from pushbridge.evaluation import make_trials, wilson
from pushbridge.precision import PrecisionFeedback, PrecisionHandoff, run_precision
from pushbridge.rl import SACPolicy
from pushbridge.sim import ORIGIN
from pushbridge.video import VideoWriter


ROOT = Path(__file__).resolve().parents[1]


def video_title(name):
    strategy = ('Contact-adapted SAC' if name.startswith('source_contact') else
                'Direct-transfer SAC' if name.startswith('source_direct') else 'Contact adapter + target RL')
    return strategy + (' + precision feedback' if name.endswith('_finish') else ' | policy alone') + ' | seed 1'


def render_observer(writer, title, trial):
    last_frame = []
    def render(env, state, phase, step):
        # The camera changes presentation only. Geometry, target radius and
        # simulation contacts are identical to the measured experiment.
        if env.renderer is None:
            env.renderer = mujoco.Renderer(env.model, height=640, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [ORIGIN[0]+.170, ORIGIN[1]+.105, .025]
        camera.distance, camera.azimuth, camera.elevation = .43, 90, -74
        env.renderer.update_scene(env.data, camera=camera)
        frame = cv2.cvtColor(env.renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (0, 0), (960, 78), (31, 41, 38), -1)
        cv2.putText(frame, title, (24, 30), cv2.FONT_HERSHEY_SIMPLEX, .67, (237, 247, 242), 1, cv2.LINE_AA)
        error = np.linalg.norm(state[2:]-trial['goal'])*1000
        text = f'{step*.1:.1f} s  |  {phase.upper()}  |  centre error {error:.1f} mm  |  full fit <= 8 mm'
        cv2.putText(frame, text, (24, 59), cv2.FONT_HERSHEY_SIMPLEX, .46, (147, 226, 192), 1, cv2.LINE_AA)
        writer.write(frame)
        last_frame[:] = [frame.copy()]
    return render, last_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/precision_finish')
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    out = args.out
    if (out/'evaluation.json').exists():
        raise SystemExit('Results already exist. Choose a new --out directory to preserve evidence.')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    checkpoints = {}
    policies = {}
    for seed in range(3):
        source = ROOT/f'results/rl_comparison/sac_demo_initialized_seed{seed}/sac.zip'
        post = ROOT/f'results/embodiment_transfer/posttrained_seed{seed}/sac.zip'
        checkpoints[str(seed)] = {'source_sha256': sha256(source), 'posttrained_sha256': sha256(post)}
        direct = SACPolicy(source)
        policies[f'source_direct_seed{seed}'] = direct
        policies[f'source_contact_seed{seed}'] = ContactAdapter(direct)
        policies[f'posttrained_seed{seed}'] = ContactAdapter(SACPolicy(post))
    protocol = {
        'question': 'Does full-object success require a different terminal objective and an explicit precision controller?',
        'followup': True, 'prior_experiment_unchanged': 'results/embodiment_transfer/evaluation.json',
        'goal_radius_mm': 20, 'puck_radius_mm': 12, 'fit_distance_mm': 8,
        'success': 'Centre distance <= 8 mm throughout ten stationary 0.1 s hold ticks after contact release for hybrid feedback; policy-only starts hold on entering the 8 mm region.',
        'decision_budget': 120, 'hold_ticks_extra_and_logged': True,
        'feedback': {'handoff_distance_mm': 30, 'stop_distance_mm': 3, 'max_action_mm': 3,
                     'push_gain': .22, 'min_push_mm': .4, 'lateral_gain': .8,
                     'release_actions': 4, 'release_step_mm': 3},
        'validation': {'seed': 7131, 'cases': 10, 'hybrid_seed1_successes': {'direct': 10, 'contact': 10},
                       'feedback_alone_successes': 8, 'note': 'One validation run; constants then frozen, no search on evaluation cases.'},
        'suites': {'paired': {'seed': 2027, 'cases': 20, 'model_seeds': [0, 1, 2]},
                   'fresh': {'seed': 9092, 'cases': 20, 'model_seeds': [1]}},
        'policy_weights': 'Unchanged. No additional imitation updates or RL training.',
        'checkpoints': checkpoints, 'training_interactions_added': 0,
        'attribution': 'Hybrid policies use learned approach then hand-coded state feedback. Include feedback-only baseline under identical budgets.',
        'limitations': ['State input from simulator, not perception or language', 'Planar circular-footprint criterion',
                        'Follow-up motivated by an observed terminal-metric mismatch', 'Fresh suite has only one model seed',
                        'Twenty paired cases per suite; training seeds are not independent task samples',
                        'Known geometry and exact state are supplied to the feedback controller'],
    }
    # Persist the complete frozen protocol before any evaluation case runs.
    write_json(out/'protocol.json', protocol)
    suites = {}
    for suite, spec in protocol['suites'].items():
        results = {}
        suite_out = out/suite
        suite_out.mkdir(exist_ok=True)
        trials = make_trials(spec['cases'], spec['seed'])
        factories = {'feedback_only': PrecisionFeedback}
        for name, policy in policies.items():
            if int(name[-1]) in spec['model_seeds']:
                factories[name] = lambda p=policy: p
                factories[name+'_finish'] = lambda p=policy: PrecisionHandoff(p)
        for name, factory in factories.items():
            rows = []
            for i, trial in enumerate(trials):
                render = suite == 'paired' and i == 0 and ('seed1' in name) and not args.no_video
                writer = VideoWriter(suite_out/f'{name}.mp4', 10, (960, 640)) if render else None
                observer, last_frame = render_observer(writer, video_title(name), trial) if writer else (None, [])
                try:
                    row, trace = run_precision(factory(), trial, observer=observer)
                    if writer:
                        # A two-second view of the actual final state, not extra
                        # simulation ticks and not part of the hold measurement.
                        for _ in range(20):
                            writer.write(last_frame[0])
                        cv2.imwrite(str(suite_out/f'{name}.jpg'), last_frame[0])
                finally:
                    if writer:
                        writer.release()
                row['trial'] = i
                rows.append(row)
                np.savez_compressed(suite_out/f'{name}_{i:03d}.npz', **trace)
            n = sum(r['success'] for r in rows)
            results[name] = {'successes': n, 'episodes': len(rows), 'wilson_95pct': wilson(n, len(rows)),
                             'mean_final_distance_mm': float(np.mean([r['final_distance_mm'] for r in rows])),
                             'mean_decisions': float(np.mean([r['decisions'] for r in rows])), 'runs': rows}
            print(f'{suite} / {name}: {n}/{len(rows)} full fits, {results[name]["mean_final_distance_mm"]:.2f} mm mean error', flush=True)
        suites[suite] = {'seed': spec['seed'], 'results': results}
    old = json.loads((ROOT/'results/embodiment_transfer/evaluation.json').read_text())['results']
    endpoints = {name: [r['final_distance_mm'] for r in old[name]['runs']] for name in policies}
    write_json(out/'evaluation.json', {'protocol': protocol, 'suites': suites,
                                      'original_endpoint_distances_mm': endpoints})
    print(f'Preserved results: {out}', flush=True)


if __name__ == '__main__':
    main()
