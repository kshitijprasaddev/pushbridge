# Longer training and a stricter objective

This follow-up keeps every original checkpoint and result. It asks whether more SAC training fixes the strict completion failures, and separates extra training from a change in the terminal success rule.

Each of the three original target SAC checkpoints received two separate continuations of 40,000 decisions. Both start at 20,000 target decisions and finish at 60,000. The first retains the original 18 mm objective. The second gives its terminal bonus only when the puck centre stays within 8 mm throughout a one-second zero-action hold. The reset distribution, dense shaping, action limits, architecture and learning rate remain the same.

The original checkpoints did not save replay data, so every continuation starts a new replay buffer. Actor, critics, entropy and optimizer states transfer. These are fixed-budget continuations, not one uninterrupted 60,000-step run. All six models use their final checkpoint. Validation on seed 7131 is diagnostic and never selects weights.

## Strict completion results

Counts below are ordered by training seeds 0, 1 and 2. Each seed faces the same 20 cases. The fresh suite uses newly declared seed 11939; the paired suite uses previously inspected seed 2027.

| Evaluation suite | Policy | Learned policy only | With the existing finisher |
| --- | --- | --- | --- |
| Paired | Original 20k target SAC | 6/20, 19/20, 16/20 | 20/20, 20/20, 19/20 |
| Paired | 60k target SAC, original objective | 18/20, 18/20, 17/20 | 20/20, 20/20, 20/20 |
| Paired | 60k target SAC, strict objective | 14/20, 19/20, 17/20 | 20/20, 20/20, 20/20 |
| Fresh | Original 20k target SAC | 9/20, 19/20, 16/20 | 20/20, 20/20, 20/20 |
| Fresh | 60k target SAC, original objective | 19/20, 19/20, 19/20 | 20/20, 20/20, 20/20 |
| Fresh | 60k target SAC, strict objective | 19/20, 18/20, 18/20 | 20/20, 20/20, 20/20 |

Policy-only uses a benchmark stop and ten zero-action hold ticks once the puck enters the 8 mm region. It does not learn a release action. The finisher is the unchanged hand-coded controller: it takes over within 30 mm, tapers the push, retracts the tool and verifies containment. Its completion must be attributed to that hybrid design.

## Inspect and reproduce

- [Frozen protocol](protocol.json): budgets, objectives, checkpoint hashes and evaluation seeds.
- [All outcomes](evaluation.json): every trial, including failures.
- [Summary](summary.json): seed comparisons and gained/lost cases.
- [Audit](audit.json): checkpoint hashes, physical traces, strict holds and preserved original baseline.
- Each model folder contains its final `sac.zip`, `training_metrics.json` and `learning_curve.json`.
- `paired/` and `fresh/` contain all 760 state, action, joint, phase and goal traces.

From the repository root after installing `.[rl,dev]`:

```bash
python scripts/rl_precision_followup.py --out results/my_rl_precision_followup --workers 3
python scripts/audit_rl_followup.py --out results/my_rl_precision_followup
python scripts/summarize_rl_followup.py --out results/my_rl_precision_followup
```

The default output is preserved evidence, so the training command refuses to overwrite it. The fresh cases remain in the same simple planar task distribution. These results use exact MuJoCo state; they do not establish visual, language, hardware or humanoid control. Longer training can improve one seed and degrade another. Every branch is retained to make that visible.
