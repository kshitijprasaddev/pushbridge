"""Fixed-budget objective-versus-budget SAC follow-up; old evidence is frozen.

Run from the repository root. Six training jobs extend unchanged target
checkpoints by 40,000 decisions each. Final checkpoints alone are evaluated;
validation curves never select weights. No human or evaluation transition is
inserted into SAC replay. Existing checkpoints contain no replay buffer.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from pushbridge.data import sha256, write_json
from pushbridge.embodiment import COMPACT_SPEC, CONTACT_SCALE, ContactAdapter
from pushbridge.evaluation import make_trials, wilson
from pushbridge.precision import PrecisionFeedback, PrecisionHandoff, run_precision
from pushbridge.precision_rl import StrictCompactGym
from pushbridge.rl import SACPolicy, observation
from pushbridge.sim import MAX_ACTION
from pushbridge.transfer import CompactGym


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
VARIANTS = ('longer_original', 'strict_objective')
ADDITIONAL_STEPS = 40000
VALIDATION_SEED = 7131
FRESH_SEED = 11939


class AgentActor:
    def __init__(self, agent):
        self.agent = agent

    def act(self, state, goal):
        action, _ = self.agent.predict(observation(state, goal), deterministic=True)
        return action * MAX_ACTION


def validate(agent):
    policy = ContactAdapter(AgentActor(agent))
    rows = [run_precision(policy, trial)[0] for trial in make_trials(10, VALIDATION_SEED)]
    return {'successes': sum(row['success'] for row in rows), 'episodes': len(rows),
            'mean_final_distance_mm': float(np.mean([row['final_distance_mm'] for row in rows]))}


class StrictCurve(BaseCallback):
    def __init__(self, out, started):
        super().__init__()
        self.out, self.started, self.rows = Path(out), started, []
        self.hold_steps = 0

    def measure(self, steps):
        row = {'additional_training_decisions': steps,
               'target_training_decisions_total': 20000 + steps,
               'new_training_hold_control_steps': self.hold_steps,
               'elapsed_seconds_including_validation': time.perf_counter() - self.started,
               **validate(self.model)}
        self.rows.append(row)
        write_json(self.out / 'learning_curve.json', {
            'validation_seed': VALIDATION_SEED, 'success_threshold_mm': 8,
            'selection': 'Final fixed-budget checkpoint only; validation never selects weights.',
            'rows': self.rows})
        print(f'{self.out.name}: +{steps} decisions | strict validation {row["successes"]}/10', flush=True)

    def _on_training_start(self):
        self.measure(0)

    def _on_step(self):
        self.hold_steps += sum(info.get('hold_control_steps', 0) for info in self.locals['infos'])
        if self.num_timesteps % 20000 == 0:
            self.measure(self.num_timesteps)
        return True


def train_job(job):
    out, variant, seed = job
    out = Path(out) / f'{variant}_seed{seed}'
    source = ROOT / f'results/embodiment_transfer/posttrained_seed{seed}/sac.zip'
    if out.exists():
        raise ValueError(f'Training output already exists: {out}')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    env = StrictCompactGym() if variant == 'strict_objective' else CompactGym()
    agent = SAC.load(source, env=env, device='cpu', seed=seed, learning_rate=.00015)
    agent.set_random_seed(seed)
    callback = StrictCurve(out, started)
    try:
        agent.learn(total_timesteps=ADDITIONAL_STEPS, callback=callback, reset_num_timesteps=True)
        agent.save(out / 'sac')
    finally:
        env.close()
    original = json.loads((source.parent / 'training_metrics.json').read_text())
    report = {'variant': variant, 'seed': seed, 'parent_checkpoint': str(source.relative_to(ROOT)),
              'parent_sha256': sha256(source), 'checkpoint_sha256': sha256(out / 'sac.zip'),
              'source_training_decisions': 40000, 'target_training_decisions_before': 20000,
              'new_target_training_decisions': int(agent.num_timesteps),
              'target_training_decisions_total': 20000 + int(agent.num_timesteps),
              'previous_target_hold_control_steps': original['terminal_hold_control_steps'],
              'new_training_hold_control_steps': callback.hold_steps,
              'new_training_control_steps': int(agent.num_timesteps) + callback.hold_steps,
              'elapsed_seconds_including_validation': time.perf_counter() - started,
              'validation_seed': VALIDATION_SEED,
              'replay': 'New empty replay; original frozen checkpoints did not include replay data.',
              'transferred': 'Actor, critics, entropy parameter and optimizer states from original20k target checkpoint.',
              'hyperparameters': {'learning_rate': .00015, 'batch_size': 128, 'train_freq': 4,
                  'gradient_steps': 1, 'buffer_size': 100000, 'learning_starts': 1000,
                  'gamma': .99, 'tau': .005, 'net_arch': [64, 64]},
              'selection': 'Final fixed-budget checkpoint; no intermediate checkpoint chosen.',
              'reward_change': ('Only terminal success predicate changes from final18mm distance to <=8mm throughout10holdticks.'
                                if variant == 'strict_objective' else 'None: original18mm terminal objective retained.')}
    write_json(out / 'training_metrics.json', report)
    return report


def protocol():
    return {
        'question': 'Is extra SAC training enough, or must the terminal objective match full containment?',
        'decision_date': '2026-10-11', 'original_evidence_preserved': True,
        'parent_checkpoints': {str(seed): {'path': f'results/embodiment_transfer/posttrained_seed{seed}/sac.zip',
            'sha256': sha256(ROOT / f'results/embodiment_transfer/posttrained_seed{seed}/sac.zip')}
            for seed in SEEDS},
        'variants': {
            'original_20k': {'new_training_decisions': 0, 'target_decisions_total': 20000,
                             'objective': 'Original18mm final hold endpoint'},
            'longer_original': {'new_training_decisions': ADDITIONAL_STEPS, 'target_decisions_total': 60000,
                                'objective': 'Unchanged18mm final hold endpoint'},
            'strict_objective': {'new_training_decisions': ADDITIONAL_STEPS, 'target_decisions_total': 60000,
                                 'objective': '8mm at entry and throughout all10zero-action holdticks'}},
        'training_seeds': list(SEEDS), 'source_training_decisions_per_model': 40000,
        'added_decisions_total': 2 * len(SEEDS) * ADDITIONAL_STEPS,
        'replay': 'Each branch starts with empty robot-physics replay; parent checkpoints omitted replay.',
        'reward': 'Same compact_potential progress -0.01; +10 stable success; -5 outside. Strict branch changes terminal success predicate only.',
        'reset_distribution': 'Unchanged CompactGym: uniform puck/goal ranges, tool40mm behind puck plus2mm noise, friction0.15/0.25/0.40.',
        'observation': 'Four contact-normalised relative-state floats from MuJoCo; no vision or language.',
        'action': 'Canonical 2D action divided by4/3 contact scale; same joint slew/bounds.',
        'embodiment': COMPACT_SPEC, 'contact_scale': CONTACT_SCALE,
        'validation': {'seed': VALIDATION_SEED, 'cases': 10, 'interval_added_decisions': 20000,
                       'use': 'Diagnostic curves only; final fixed checkpoint, no selection or tuning.'},
        'suites': {'paired': {'seed': 2027, 'cases': 20, 'status': 'Previously inspected cases, retained for paired comparison'},
                   'fresh': {'seed': FRESH_SEED, 'cases': 20, 'status': 'Newly declared seed, not previously used for evaluation or tuning'}},
        'evaluation': {'model_seeds': list(SEEDS), 'modes': ['policy_only', 'with_existing_finisher'],
                       'decision_budget': 120, 'fit_distance_mm': 8, 'hold_ticks': 10,
                       'baselines': ['Original20k target SAC', 'Feedback-only controller'],
                       'hold': 'Ten zero-action physics ticks; all saved states remain inside. Extra holdticks counted separately.',
                       'release': 'Explicit hand-coded release only in finisher and feedback baseline; policy-only uses automatic benchmark hold.'},
        'selection': 'Train all six final fixed-budget models before evaluating paired or fresh cases; retain every result including regressions.',
        'limitations': ['Small20case suites; repeated models are not independent task cases',
                        'Fresh suite is an in-distribution sample, not new tasks or embodiments',
                        'Exact simulator state; no perception, language, hardware or VLA claim',
                        'Strict objective was motivated after seeing original failures',
                        'Finisher remains engineered state feedback; zero-action hold is benchmark logic',
                        'Additional budget alone cannot guarantee improvement; all outcomes reported'],
    }


def evaluate(out, spec):
    policies = {}
    for seed in SEEDS:
        policies[f'original_20k_seed{seed}'] = ContactAdapter(SACPolicy(ROOT / f'results/embodiment_transfer/posttrained_seed{seed}/sac.zip'))
        for variant in VARIANTS:
            policies[f'{variant}_seed{seed}'] = ContactAdapter(SACPolicy(out / f'{variant}_seed{seed}/sac.zip'))
    suites = {}
    for suite, suite_spec in spec['suites'].items():
        suite_out = out / suite
        suite_out.mkdir()
        factories = {'feedback_only': PrecisionFeedback}
        for name, policy in policies.items():
            factories[name] = lambda policy=policy: policy
            factories[name + '_finish'] = lambda policy=policy: PrecisionHandoff(policy)
        results = {}
        for name, factory in factories.items():
            rows = []
            for i, trial in enumerate(make_trials(suite_spec['cases'], suite_spec['seed'])):
                row, trace = run_precision(factory(), trial)
                row['trial'] = i
                rows.append(row)
                np.savez_compressed(suite_out / f'{name}_{i:03d}.npz', **trace)
            n = sum(row['success'] for row in rows)
            results[name] = {'successes': n, 'episodes': len(rows), 'wilson_95pct': wilson(n, len(rows)),
                             'mean_final_distance_mm': float(np.mean([row['final_distance_mm'] for row in rows])),
                             'mean_decisions': float(np.mean([row['decisions'] for row in rows])), 'runs': rows}
            print(f'{suite}/{name}: {n}/{len(rows)} strict fits', flush=True)
        suites[suite] = {'seed': suite_spec['seed'], 'results': results}
        write_json(out / 'evaluation_progress.json', {'protocol': spec, 'suites': suites})
    write_json(out / 'evaluation.json', {'protocol': spec, 'suites': suites})
    return suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/rl_precision_followup')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Output already exists; choose a new directory to preserve frozen evidence.')
    args.out.mkdir(parents=True)
    spec = protocol()
    # This record is written before training or any new evaluation cases run.
    write_json(args.out / 'protocol.json', spec)
    jobs = [(str(args.out), variant, seed) for seed in SEEDS for variant in VARIANTS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(train_job, jobs))
    write_json(args.out / 'training_summary.json', {'models': reports,
        'new_training_decisions_total': sum(row['new_target_training_decisions'] for row in reports),
        'new_training_control_steps_total': sum(row['new_training_control_steps'] for row in reports)})
    torch.set_num_threads(2)
    evaluate(args.out, spec)
    print(f'Frozen follow-up evidence: {args.out}', flush=True)


if __name__ == '__main__':
    main()
