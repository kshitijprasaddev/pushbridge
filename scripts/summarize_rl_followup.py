"""Summarise every fixed-budget follow-up branch, including regressions."""
import argparse
import json
from pathlib import Path

from pushbridge.data import write_json


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('original_20k', 'longer_original', 'strict_objective')
LABELS = {'original_20k': 'Original 20k target SAC',
          'longer_original': '60k target SAC, original objective',
          'strict_objective': '60k target SAC, strict objective'}


def summarize(out):
    evaluation = json.loads((out / 'evaluation.json').read_text())
    training = json.loads((out / 'training_summary.json').read_text())
    summary = {'question': evaluation['protocol']['question'],
               'training_decisions_added': training['new_training_decisions_total'],
               'training_control_steps_added': training['new_training_control_steps_total'],
               'baseline_target_decisions': 20000, 'followup_target_decisions': 60000,
               'seed_order': [0, 1, 2], 'suites': {},
               'interpretation': [
                   'The old training objective accepted an 18 mm centre error. Full containment requires at most 8 mm.',
                   'Longer-original isolates extra training from a change to the terminal objective.',
                   'Strict-objective has the same added budget and changes only the terminal success predicate.',
                   'All branches load the original frozen target checkpoint, then start a new replay buffer.',
                   'Policy-only still uses the benchmark zero-action hold after entering the containment region.',
                   'The finisher uses engineered state feedback and release actions, so its successes are not learned completion alone.',
                   'The paired suite was previously inspected; fresh seed 11939 was declared before these runs.',
                   'Training seeds share the same 20 task cases. Sixty model-case runs are not 60 independent tasks.']}
    lines = ['# Longer training and a stricter objective', '',
             'This follow-up keeps every original checkpoint and result. It asks whether more SAC training fixes the strict completion failures, and separates extra training from a change in the terminal success rule.', '',
             'Each of the three original target SAC checkpoints received two separate continuations of 40,000 decisions. Both start at 20,000 target decisions and finish at 60,000. The first retains the original 18 mm objective. The second gives its terminal bonus only when the puck centre stays within 8 mm throughout a one-second zero-action hold. The reset distribution, dense shaping, action limits, architecture and learning rate remain the same.', '',
             'The original checkpoints did not save replay data, so every continuation starts a new replay buffer. Actor, critics, entropy and optimizer states transfer. These are fixed-budget continuations, not one uninterrupted 60,000-step run. All six models use their final checkpoint. Validation on seed 7131 is diagnostic and never selects weights.', '',
             '## Strict completion results', '',
             'Counts below are ordered by training seeds 0, 1 and 2. Each seed faces the same 20 cases. The fresh suite uses newly declared seed 11939; the paired suite uses previously inspected seed 2027.', '',
             '| Evaluation suite | Policy | Learned policy only | With the existing finisher |',
             '| --- | --- | --- | --- |']
    for suite, suite_data in evaluation['suites'].items():
        results = suite_data['results']
        suite_summary = {'seed': suite_data['seed'], 'cases': 20, 'variants': {}}
        for variant in VARIANTS:
            policies = [results[f'{variant}_seed{seed}'] for seed in range(3)]
            hybrid = [results[f'{variant}_seed{seed}_finish'] for seed in range(3)]
            alone_counts = [row['successes'] for row in policies]
            hybrid_counts = [row['successes'] for row in hybrid]
            base = [results[f'original_20k_seed{seed}']['runs'] for seed in range(3)]
            paired_changes = []
            failures = []
            for seed, result in enumerate(policies):
                gains, losses = [], []
                for baseline, row in zip(base[seed], result['runs']):
                    if row['success'] and not baseline['success']:
                        gains.append(row['trial'])
                    if baseline['success'] and not row['success']:
                        losses.append(row['trial'])
                    if not row['success']:
                        failures.append({'model_seed': seed, 'trial': row['trial'],
                                         'final_distance_mm': row['final_distance_mm']})
                paired_changes.append({'model_seed': seed, 'gained_success_cases': gains,
                                       'lost_success_cases': losses})
            suite_summary['variants'][variant] = {
                'label': LABELS[variant], 'policy_only_successes_by_seed': alone_counts,
                'finisher_successes_by_seed': hybrid_counts,
                'policy_only_mean_final_distance_mm_by_seed': [r['mean_final_distance_mm'] for r in policies],
                'changes_vs_original_per_model_seed': paired_changes,
                'policy_only_failures': failures}
            lines.append(f'| {suite.title()} | {LABELS[variant]} | {", ".join(f"{n}/20" for n in alone_counts)} | {", ".join(f"{n}/20" for n in hybrid_counts)} |')
        suite_summary['feedback_only'] = {key: results['feedback_only'][key]
                                          for key in ('successes', 'episodes', 'mean_final_distance_mm')}
        summary['suites'][suite] = suite_summary
    lines += ['', 'Policy-only uses a benchmark stop and ten zero-action hold ticks once the puck enters the 8 mm region. It does not learn a release action. The finisher is the unchanged hand-coded controller: it takes over within 30 mm, tapers the push, retracts the tool and verifies containment. Its completion must be attributed to that hybrid design.', '',
              '## Inspect and reproduce', '',
              '- [Frozen protocol](protocol.json): budgets, objectives, checkpoint hashes and evaluation seeds.',
              '- [All outcomes](evaluation.json): every trial, including failures.',
              '- [Summary](summary.json): seed comparisons and gained/lost cases.',
              '- [Audit](audit.json): checkpoint hashes, physical traces, strict holds and preserved original baseline.',
              '- Each model folder contains its final `sac.zip`, `training_metrics.json` and `learning_curve.json`.',
              '- `paired/` and `fresh/` contain all 760 state, action, joint, phase and goal traces.', '',
              'From the repository root after installing `.[rl,dev]`:', '',
              '```bash',
              'python scripts/rl_precision_followup.py --out results/my_rl_precision_followup --workers 3',
              'python scripts/audit_rl_followup.py --out results/my_rl_precision_followup',
              'python scripts/summarize_rl_followup.py --out results/my_rl_precision_followup',
              '```', '',
              'The default output is preserved evidence, so the training command refuses to overwrite it. The fresh cases remain in the same simple planar task distribution. These results use exact MuJoCo state; they do not establish visual, language, hardware or humanoid control. Longer training can improve one seed and degrade another. Every branch is retained to make that visible.', '']
    write_json(out / 'summary.json', summary)
    (out / 'README.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/rl_precision_followup')
    args = parser.parse_args()
    summary = summarize(args.out)
    for suite, data in summary['suites'].items():
        for variant, row in data['variants'].items():
            print(f'{suite}/{variant}: policy {row["policy_only_successes_by_seed"]}; finisher {row["finisher_successes_by_seed"]}')


if __name__ == '__main__':
    main()
