# Experiment details

These sections describe the original 18 mm centre-distance benchmarks. The separate [precision evaluation](PRECISION.md) uses an 8 mm whole-object criterion. The [longer SAC comparison](../results/rl_precision_followup/README.md) continues each original compact-arm checkpoint for another 40,000 decisions under two terminal objectives. Both additions leave the original results unchanged. Start with the [reviewer guide](REVIEWER_GUIDE.md) for a shorter tour.

**Five personal phone recordings → contact-space policies → two robot embodiments in MuJoCo.**

The research question: can a small embodiment adapter preserve useful human demonstration knowledge when the robot's joints, tip geometry and dynamics change?

The recordings train a contact dynamics ensemble and supply actor action labels. We compare personal-data post-training, imitation initialization followed by SAC, learned-model MPC, classical feedback, and transfer to a compact three-joint arm. A separate inference experiment exports the same trained actor and checks its physical trajectories.

**[Open the interactive research report](../index.html).** It runs entirely from included HTML, CSS, JavaScript, metrics and videos. No Node build or hosted service is needed. The experiment uses the challenge's world-model route; it is **state-based robot learning, not a VLA**.

## Run the submission

Open `index.html` directly after extracting the project, or serve the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis,rl]'
pushbridge serve --port 8767
# Open http://127.0.0.1:8767/
```

The report and saved results are immediately usable. Training requires CPU only. On the original workspace, `./run-local.sh serve --port 8767` uses the already-installed runtime. If the port is occupied, choose another. On Windows, activate `.venv\Scripts\activate`.

Linux headless rendering defaults to EGL. If unavailable, use a working OSMesa setup with `MUJOCO_GL=osmesa`, or a desktop OpenGL context with `MUJOCO_GL=glfw`. MuJoCo version is pinned to 3.3.7. Python 3.12 was tested; package requirements allow Python 3.10+.

## What the personal data contributes

Five user-supplied recordings are preserved in `data/raw/personal_circle/` as publication copies. Audio and GPS/device metadata were removed with a lossless video remux; every decoded video frame is verified identical. The extraction manifest retains original and publication container hashes, plus the matching decoded-frame hashes. The declared collector is the applicant; software cannot verify authorship. Red object and blue fingertip markers yield 289 usable transitions, including 195 with object motion. Whole recordings are split, rather than adjacent frames:

| Split | Clips | Use |
| --- | --- | --- |
| Training | `Video`, `Video_2`, `Video_3` | Dynamics fitting, demonstration retrieval, actor labels |
| Validation | `Video_1` | World-model selection and empirical error-radius calibration |
| Test | `IMG_5670` | Held-out prediction evaluation only |

Circle-relative coordinates compensate approximate image translation and uniform scale changes. The observed circle radius maps to a **chosen 20 mm simulator radius**, not measured real-world distances. This does not recover full camera perspective or rotation. Occluded tracking segments are excluded, never bridged. Raw hashes and original frame timestamps are recorded in the manifest.

## 1. Post-train an existing policy on personal recordings

After scratch SAC trains for 40,000 simulator decisions, fit its actor to training-only fingertip action labels for **200 supervised updates at learning rate 0.0001**. There are no additional simulator training interactions. Critics remain unchanged.

| Training seed | Before personal post-training | After |
| --- | --- | --- |
| 0 | 13/20 | 20/20 |
| 1 | 9/20 | 5/20 |
| 2 | 20/20 | 19/20 |

One seed improves and two decline. Human action labels can conflict with learned robot behavior; supervised fitting is not automatically an improvement. Source pretraining here is simulator SAC, not an external pretrained VLA. [Protocol](../results/personal_posttraining/protocol.json) · [Videos and outcomes](../results/personal_posttraining/evaluation/report.html)

## 2. Bootstrap with demonstrations, then use RL

Initialize actor means with the three personal training clips, then train SAC through physical robot interaction. The dataset contributes 123 filtered original action labels, augmented by consistent rotations. Human transitions **never enter the robot critic or replay buffer**.

| Source 2R policy | Seed 0 / 1 / 2 successes | Mean final simulator distance |
| --- | --- | --- |
| Personal-data imitation only | 16/20 · 11/20 · 18/20 | 32.2 mm |
| SAC from scratch | 13/20 · 9/20 · 20/20 | 17.6 mm |
| Personal initialization → SAC | 20/20 · 20/20 · 20/20 | 15.6 mm |
| Classical feedback | 20/20 | 13.8 mm |
| Learned-model MPC | 20/20 | 12.8 mm |

Each SAC model receives 40,000 simulator decisions. Terminal holds add physical control ticks, logged separately. Final fixed-budget checkpoints are evaluated; validation curves do not select the best weights. Dense reward shaping explicitly supplies goal progress and a behind-object contact pose. Early SAC updates forget some initialization behavior before recovery, so the experiment does not establish consistently faster learning. MPC and classical control already solve this source task without simulator policy training. PPO and TD3 were not trained. [Original study](../results/rl_comparison/report.html)

## 3. Adapt to a different embodiment

| Property | Source 2R | Compact 3R |
| --- | --- | --- |
| Link lengths | 320 + 300 mm | 240 + 170 + 130 mm |
| Tip radius | 8 mm | 3 mm |
| Puck mass | 25 g | 50 g |
| Joint-command speed limit | No added slew limit | 0.25 rad/s |
| Tool target realization | Analytic two-joint IK | Bounded damped least squares + null-space posture preference |

A **contact-radius adapter** multiplies tool-object displacement by `(12 + 8)/(12 + 3) = 20/15`, then undoes this scaling for actions. It preserves a familiar contact clearance for the source policy. This engineered map is a heuristic, not a guarantee of dynamics equivalence. Joint bounds and command slew limits constrain inverse kinematics; object motion still arises solely from MuJoCo contact, gravity and friction.

| Compact 3R method | Seed 0 / 1 / 2 successes | Mean final distance |
| --- | --- | --- |
| Source SAC, direct transfer | 19/20 · 0/20 · 9/20 | 18.5 mm |
| Frozen SAC + contact adapter | 20/20 · 20/20 · 20/20 | 16.2 mm |
| Adapter + 20,000 target RL decisions | 20/20 · 20/20 · 19/20 | 17.6 mm |
| Source classical feedback | 0/20 | 134.8 mm |
| Source feedback + adapter | 1/20 | 47.9 mm |
| Compact-aware classical feedback* | 14/20 | 18.9 mm |
| Learned-model MPC + adapter | 17/20 | 18.0 mm |

*The compact-aware baseline was added after the initial comparison and inspected on validation tasks. It uses slower pushes and tighter lateral feedback appropriate to the narrow tip. Original results remain included.

**Retargeting helps; extra target RL does not improve this suite.** Direct-transfer seed 1 stalls close to the success threshold, so the apparently large success-count change should be read alongside the modest distance change. Target post-training seed 2 creates a substantial failure in trial 3, ending 111.6 mm from the goal. Both failure replays exactly reproduce recorded physics states. [Protocol](../results/embodiment_transfer/protocol.json) · [All trials](../results/embodiment_transfer/evaluation.json)

## 3a. Extend target training under two terminal objectives

The later [longer-training comparison](../results/rl_precision_followup/README.md) tests whether budget or the original 18 mm training objective explains strict misses. Each original 20k compact-arm checkpoint receives two sibling +40k continuations, keeping the old objective or training for 8 mm with a full ten-tick hold. Hyperparameters and reset distribution stay the same. The original checkpoints contain no replay buffer, so both branches start with an empty buffer while preserving actor, critic and optimizer state.

Final fixed-budget weights are evaluated on the original paired seed 2027 and newly declared fresh seed 11939, with and without the unchanged finisher. Learned-only paired counts change from 6/19/16 to 18/18/17 with the original objective and 14/19/17 with the strict objective. Fresh counts are 9/19/16, 19/19/19 and 19/18/18 respectively. Every count has denominator 20. Both new hybrids achieve 20/20 in all seeds and both suites; learned-only completion still fails, and the strict objective is not universally better. The 760 new traces, six final weights, validation curves and budget accounting are included with a separate audit.

## 4. Inspect action-conditioned state predictions

The three-member world model is trained exclusively on personal tracks. It learns displacement gains and contact range inside an explicitly engineered rotation-equivariant contact structure.

| Open-loop horizon | Learned error | Fixed-contact baseline | No-motion baseline | Test windows |
| --- | --- | --- | --- | --- |
| 0.1 s | 0.89 mm | 1.22 mm | 1.64 mm | 111 |
| 0.5 s | 4.06 mm | 5.49 mm | 7.66 mm | 22 |
| 1.0 s | 7.92 mm | 10.30 mm | 14.96 mm | 11 |
| 2.0 s | 13.80 mm | 17.36 mm | 28.92 mm | 5 |
| 4.0 s | 21.82 mm | 32.94 mm | 48.51 mm | 2 |

Errors use the **normalised simulator scale** and one held-out recording. Future finger actions are supplied; the object state is not corrected during each rollout. The dashboard projects predictions onto the corresponding original video frames, with preserved source timestamps. This is a state-prediction overlay, **not generative video prediction**.

The frozen model also has roughly 0.86 mm one-step mean error on source and compact robot traces, using achieved tool displacement. These domain diagnostics are not training data. Validation-calibrated one-step radii have no formal coverage guarantee. [Windows, errors and domain diagnostics](../results/world_model_visuals/metrics.json)

## 5. Optimize the trained actor

Export the deterministic 4 → 64 → 64 → 2 mean actor into an 18 KB float32 weight file. Compare the standard SB3 prediction path against direct Torch kernels and NumPy matrix multiplies. Complete action requests include observation encoding and norm clipping.

| Path | Median request | p95 request | Relative median speed |
| --- | --- | --- | --- |
| Standard SB3 | 0.175 ms | 0.229 ms | 1× |
| Direct Torch actor | 0.045 ms | 0.060 ms | **3.92×** |
| NumPy actor | 0.023 ms | 0.031 ms | **7.63×** |

Direct Torch reproduces **bit-identical actions and states in all 20 physical rollouts**. NumPy preserves all 20 success outcomes, but nanometre-scale action differences can accumulate into up to 5.7 mm of tool/object state-vector difference through contact and threshold timing. The latter is not an exact-equivalence deployment path.

These are local CPU measurements: 100 warm-up calls, 3,000 timed calls per method, alternating paired blocks. They are not edge-hardware claims or total robot-loop speedups. [Exact measurements and trace checks](../results/policy_speed/metrics.json)

## Reproduce the experiments

Run from the project directory. Saved checkpoints and results are included; use fresh output folders when retraining.

```bash
# Re-extract the applicant recordings and fit the dynamics.
pushbridge ingest-circle --videos data/raw/personal_circle \
  --out data/processed/rerun --collector "Your name"
pushbridge train --dataset data/processed/rerun --out results/rerun/model --seed 7

# Source SAC study (six training runs).
pushbridge rl-experiment --dataset data/processed/personal_five \
  --checkpoint results/personal_five/model/world_model.pt \
  --out results/rl_rerun --steps 40000 --seeds 0 1 2
python scripts/summarize_rl.py --run results/rl_rerun

# Personal-data post-training, no new simulator training.
python scripts/posttrain_personal.py --source-root results/rl_comparison \
  --out results/posttraining_rerun

# Compact embodiment, three 20k adaptation runs + paired evaluation.
pushbridge transfer-experiment --source-root results/rl_comparison \
  --out results/transfer_rerun --steps 20000 --seeds 0 1 2

# Actor export, timing and physical trace agreement.
python scripts/benchmark_actor.py --out results/speed_rerun

# Rebuild visual evidence/dashboard using the included reference results.
python scripts/build_world_visuals.py
python scripts/build_dashboard.py
pytest -q
```

The source runs consumed 240,000 training decisions; compact adaptation adds 60,000. Extra terminal-hold ticks and validation rollouts are recorded separately. The original six source runs took about 7.6 minutes total on CPU; the three target runs took about 2.1 minutes, including validation. Timing depends on hardware. Supervised fitting uses no new simulator training decisions.

## Design choices and limits

- A task-space policy isolates the hand-to-tool contact problem from robot joint layout. The compact arm remains planar; it is not a humanoid or full 3D manipulator.
- Markers make the tiny personal dataset inspectable. Simulator controllers receive exact positions, so this is not an end-to-end vision policy or a VLA.
- Whole-clip splits prevent adjacent-frame leakage. Five recordings from one object/surface and three training seeds remain a small experiment.
- Each model seed faces the same twenty task cases. Repetition does not create sixty independent task samples. Source and target suites use different declared seeds; compare methods within each suite.
- Geometric model structure, MPC candidates, dense rewards, known contact radii and IK are engineered knowledge. The learned contribution and its limitations are explicit.
- MPC and classical feedback tie source-arm success. Personal supervised post-training and additional target RL have mixed or negative effects. Those outcomes are retained.

See [design details](DESIGN.md), [validation](VALIDATION.md), and [the home recording guide](RECORDING.md). Earlier synthetic smoke checks and the hand-coded single-video follower are archived separately; they are not learned-policy results.

The project is published at [kshitijprasaddev/pushbridge](https://github.com/kshitijprasaddev/pushbridge), with a [live research app](https://pushbridge.vercel.app/). Application submission is handled by the applicant. Code is MIT licensed; the applicant controls the recording rights.
