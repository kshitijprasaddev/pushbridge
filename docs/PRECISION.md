# Precision: correcting the terminal objective

The original evaluator and SAC reward used a centre-distance success threshold of 18 mm. That metric was useful for initial contact transfer, but its success label did not mean the whole object fitted inside the visual goal.

For circular planar footprints, containment is exact:

```text
clearance = goal_radius - puck_radius - distance(puck_centre, goal_centre)
          = 20 mm - 12 mm - centre_error

full containment: clearance >= 0, equivalently centre_error <= 8 mm
```

This ignores the object's vertical height and applies to its ground-plane footprint. Real recording dimensions were not measured; millimetres refer to the chosen simulator scale.

## Two distinct interventions

**Policy alone:** reuse the exact checkpoint and adapter, change the evaluator to continue until it enters the 8 mm region, then require all ten stationary hold ticks to remain inside. The historical evaluation stopped at 18 mm, so this is a new run, not a new score assigned to an old video.

**Policy plus feedback:** use the same learned approach, then permanently hand over at centre distance 30 mm. Classical feedback aligns behind the puck, applies pushes of at most 3 mm with tighter lateral correction, and reduces push size proportionally to remaining distance. At 3 mm it retracts four times by 3 mm along the opposite push direction, then verifies the stationary hold.

The explicit controller receives known tool/object radii and exact state. No new imitation, RL training, target enlargement or object teleporting occurs. Commanded joint speed remains limited to 0.25 rad/s. Different actor paths can approach the handoff with different contact orientations.

Both versions have 120 policy/control decisions. Ten-tick hold attempts add physics ticks outside that decision budget, matching the original accounting convention. All extra hold ticks are logged. Failed attempts can continue within the decision budget. Feedback-only uses the same alignment/push/release controller from the initial state.

The feedback release requirement is part of the hybrid procedure. Policy-only has no release controller and starts verification when it enters the region. Consequently, a feedback-only run can end geometrically inside but fail to finish its release and hold before the budget expires. The hold and decision counts distinguish this from missing the target.

## Protocol and chronology

This follow-up was motivated by the observed mismatch in the original report. It was not prespecified before that report. One 10-case validation run used seed 7131: seed 1 direct and adapted hybrids each solved 10/10; feedback alone solved 8/10. Constants were then frozen and written to [protocol.json](../results/precision_finish/protocol.json) before evaluation.

The paired suite repeats the original 20 cases (seed 2027) for all three checkpoints. An additional fresh suite uses 20 cases from seed 9092 with seed 1 checkpoints. No feedback settings or checkpoint selection changed after evaluation. The supplied checkpoint hashes are recorded in the protocol.

## Results

| Approach | Policy alone | With feedback | Mean error with feedback, seeds 0 / 1 / 2 |
| --- | --- | --- | --- |
| Direct transfer | 0/20 · 0/20 · 0/20 | 20/20 · 20/20 · 20/20 | 2.69 · 2.76 · 2.74 mm |
| Contact adapter | 10/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 | 2.68 · 2.76 · 2.72 mm |
| Adapter + RL | 6/20 · 19/20 · 16/20 | 20/20 · 20/20 · 19/20 | 2.70 · 2.73 · 8.15 mm |

Feedback alone solves 18/20 in both paired and fresh suites. Its mean decision counts are 99.55 and 100.15; failed cases can have small final error without completing release and verification in time. All seed 1 hybrids solve 20/20 fresh cases; policy-only counts are 0/20, 20/20 and 20/20 respectively.

The RL seed 2 hybrid failure is paired trial 3. The policy never reaches the 30 mm handoff region, so feedback is never activated. The mean includes this failure. This controller is not a general recovery method for a failed approach.

Wilson intervals are included per 20-case run. Repeating task cases across training seeds does not create 60 independent samples. The fresh check evaluates one checkpoint seed in the same reset/friction distribution, not a novel real-world embodiment.

## Evidence and reproduction

Every trial has saved states, tool actions, joint angles, phase labels and a goal. Successful holds and speed limits can be audited numerically:

```bash
python scripts/audit_submission.py
python scripts/precision_experiment.py --out results/precision_rerun
```

The six native videos show paired task 0 for seed 1, with and without feedback for all three strategies. Their final two seconds repeat the actual final image for readability; those frames are not extra simulation time. The task inspector reconstructs arm joints from the same saved angles, plots object position at its true scale, and does not interpolate between recorded states.

The tolerance audit uses original final distances. It is a post-hoc endpoint sensitivity check, not evidence that the old trajectories passed a different stable-hold criterion. [All outcomes and frozen protocol](../results/precision_finish/evaluation.json)
