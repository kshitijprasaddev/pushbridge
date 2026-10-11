# PushBridge

[![Open the live PushBridge research app: 3D robot replays, MuJoCo footage and policy comparisons](docs/assets/open-demo.svg)](https://pushbridge.vercel.app/)

### [Open the live app → pushbridge.vercel.app](https://pushbridge.vercel.app/)

Start here to explore the project. **No installation or sign-in required.** Replay a robot push, compare the learned policy with precision feedback, and watch the native MuJoCo recordings. The app also opens the methods and results in a readable evidence viewer.

[Take the two-minute tour](https://pushbridge.vercel.app/?evidence=%2Fevidence%2Fdocs%2FREVIEWER_GUIDE.md) · [Watch MuJoCo footage](https://pushbridge.vercel.app/#mujoco) · [Read the experiment methods](https://pushbridge.vercel.app/?evidence=%2Fevidence%2Fdocs%2FEXPERIMENTS.md)

**From five home videos to contact-aware robot control in MuJoCo.**

A finger pushes a red cap into a drawn circle. This project extracts those motions, uses them to initialize a learned policy, and transfers the policy to a robot with different joints, tool geometry and object dynamics. It also learns object-motion predictions and exports a faster actor.

The interactive app uses React, Vite and Three.js. The source, saved checkpoints and reproduction steps are included below. For local setup and deployment, see [the web app guide](docs/WEB_APP.md).

[![Open the interactive app: Three.js replay of a recorded MuJoCo task](web/preview.jpg)](https://pushbridge.vercel.app/)

## Run locally

The saved results are included. You can build the app without installing Python or retraining the policies. Use Node 24.x. These commands select the pinned package manager, so a global pnpm installation is optional:

```bash
cd web
npx --yes pnpm@11.25.0 install --frozen-lockfile
npx --yes pnpm@11.25.0 dev
# Open the local URL printed by Vite.
```

Select a transfer strategy, switch between policy alone and policy plus feedback, then replay or scrub a task. The 3D arm follows recorded joint angles; the object follows recorded positions. Three.js renders the scene, while MuJoCo produced the underlying physics traces. The browser does not run a new policy rollout.

The [MuJoCo gallery](https://pushbridge.vercel.app/#mujoco) shows recordings from the original physics renderer. The evidence library opens documents as articles and trial JSON as a searchable table, with the original files available to download. A training seed identifies a model; a task identifies one of the shared starting conditions. Scores count 20 tasks, while each replay shows one task.

For a production build, run `npx --yes pnpm@11.25.0 build` and `npx --yes pnpm@11.25.0 preview` from `web/`. The earlier [offline report](index.html) and all experiment reports remain in the repository for direct inspection.

To host the app on Vercel, import this repository and set **Root Directory to `web`**. The included `web/vercel.json` selects Vite, the pinned pnpm install and build commands, and the `dist` output directory. Node 24.x is selected by `web/package.json`. The exported replay data, videos and downloads are already inside `web/public`; the deployment needs no Python environment, API keys or GPU service. See [the deployment steps](docs/WEB_APP.md#deploy-to-vercel).

## What is actually learned?

| Component | Input and role |
| --- | --- |
| Personal demonstrations | Five original clips; three train, one validates, one tests. 289 usable transitions. |
| SAC actor | Fingertip actions initialize its mean action; simulator interactions train the robot policy. |
| World model | Personal tracks train an ensemble predicting object displacement from contact and tool actions. |
| Embodiment adapter | Known tool radii transform contact-relative observations and actions. |
| Precision feedback | Explicit classical control handles the last 30 mm, then releases contact. No learned weights. |
| Exported actor | Same policy weights with less inference overhead. |

The robot policy takes **simulator positions**, not images or language. The world model predicts **states**, with future tool actions supplied. This is a state-based robot-learning study, not an end-to-end VLA or generative video model.

## Define a complete push

The original success rule was a centre error below 18 mm. The target radius is 20 mm; the puck radius is 12 mm. To fit the entire puck, the centre error must be **at most 20 − 12 = 8 mm**. At 18 mm, the puck can protrude 10 mm outside the target.

The stricter evaluation removes that early stop and requires full containment throughout a one-second stationary hold. It uses the same learned checkpoints. A second comparison adds the same hand-coded precision feedback to each approach policy, so its contribution can be measured separately.

| Compact 3R approach | Policy alone, strict rule | With precision feedback, strict rule |
| --- | --- | --- |
| Direct transfer | 0/20 · 0/20 · 0/20 | **20/20 · 20/20 · 20/20** |
| Contact adapter | 10/20 · 20/20 · 20/20 | **20/20 · 20/20 · 20/20** |
| Adapter + target RL | 6/20 · 19/20 · 16/20 | **20/20 · 20/20 · 19/20** |

Cells list training seeds 0 / 1 / 2 on the same 20 paired cases. They are not 60 independent tasks. Direct and contact-adapted hybrids finish at roughly 2.7 mm mean centre error. The RL hybrid still has one failure; its three-seed mean is 4.5 mm including that failure.

**Control attribution matters.** Feedback alone solves 18/20 under the same 120-decision budget. Some adapted policies already finish when allowed to continue; the finisher improves precision rather than creating every success. On 20 fresh tasks with seed 1 checkpoints, all three hybrids solve 20/20; policy-only direct transfer remains 0/20. [Protocol and all individual outcomes](results/precision_finish/evaluation.json) · [Precision methodology](docs/PRECISION.md)

## Does longer RL training finish the task?

The original compact-arm RL policy was trained to stop within **18 mm**, while strict completion requires **8 mm for a full second**. Two controlled continuations give every original seed another 40,000 SAC decisions, reaching 60,000 target decisions: one keeps the original objective; one tightens the training goal to 8 mm and checks every hold tick. They use the final fixed-budget checkpoints, with no selection on test performance.

| Policy alone, strict completion | Original paired tasks, seeds 0 / 1 / 2 | New fresh tasks, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Original 20k, 18 mm training goal | 6/20 · 19/20 · 16/20 | 9/20 · 19/20 · 16/20 |
| Longer 60k, same training goal | 18/20 · 18/20 · 17/20 | 19/20 · 19/20 · 19/20 |
| Strict 60k, 8 mm + hold training goal | 14/20 · 19/20 · 17/20 | 19/20 · 18/20 · 18/20 |

More training improves the weakest seed substantially, but the learned policy still has failures. The stricter objective is not uniformly better. Both new branches with the unchanged precision controller complete **20/20 for every seed on both suites**; that final control contribution remains explicit. The old missed seed 2 Task 04 moves from 111.6 mm error to 11.2 mm under the longer policy, then to 2.8 mm with feedback. The policy alone still misses that finish.

The follow-up adds 240,000 training decisions and 760 evaluated traces. All 120 paired original-baseline traces remain bit-identical. Original results and checkpoints stay available. [Protocol, all results and reproduction](results/rl_precision_followup/README.md) · [Follow-up audit](results/rl_precision_followup/audit.json)

## Other findings

| Experiment | Result and practical limit |
| --- | --- |
| Personal initialization + SAC | 20/20 for all three source-arm seeds under the original 18 mm rule. Classical control and MPC also solve 20/20. |
| Personal post-training after SAC | Improves one seed and worsens two. Human action labels can conflict with robot behavior. |
| Contact-space transfer | Original compact-arm frozen adapter solves 20/20 for all seeds at 18 mm. More target RL introduces one failure. |
| State world model | 0.89 mm one-step mean error; 7.92 mm at ten steps, on one held-out recording and a chosen simulator scale. |
| Actor inference | Direct Torch median 3.92× faster, with identical actions and states in all 20 tested rollouts. Local CPU result. |

SAC means Soft Actor-Critic, the reinforcement-learning algorithm used here. MPC means Model Predictive Control: score candidate actions using predicted future object motion. [Full metrics, budgets and reproduction commands](docs/EXPERIMENTS.md)

## Install and reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis,rl]'
python -m pytest -q
python scripts/audit_submission.py
python scripts/audit_rl_followup.py
python scripts/precision_experiment.py --out results/precision_rerun
```

Run commands from the repository root. New results go to a fresh folder, preserving included evidence. The precision comparison adds no training; it loads the included checkpoints. Full retraining commands are in [EXPERIMENTS.md](docs/EXPERIMENTS.md). `make help` lists common tasks. `make package` produces a CRC-checked ZIP containing the files selected for GitHub, excluding caches and ignored exploratory runs.

Python 3.12 and CPU execution were tested. MuJoCo is pinned to 3.3.7. Linux rendering defaults to EGL; a working OSMesa setup can use `MUJOCO_GL=osmesa`. Windows virtual environments activate via `.venv\Scripts\activate`.

## Repository map

```text
pushbridge/
├── README.md                 # Start here: question, results, setup
├── web/                      # React + Three.js app, lockfile and public evidence
├── index.html                # Earlier offline research report
├── dashboard/                # Earlier UI and complete generated report data
├── src/pushbridge/           # Tracking, models, policies, physics and controllers
├── scripts/                  # Experiments, artifact builders and evidence audit
├── tests/                    # Contact, tracking, kinematics and deployment checks
├── data/raw/personal_circle/ # Five recordings with private metadata removed
├── data/processed/           # Extracted tracks, splits and tracking previews
├── results/                  # Protocols, checkpoints, traces, metrics and videos
├── docs/                     # Reviewer guide, methodology and validation
└── .github/workflows/        # CPU tests for the package
```

All object motion after reset comes from contact physics. Known geometry, exact simulator state, dense reward shaping, a small dataset and planar tasks limit the conclusions. The detailed reports include failures, whole-clip splits and resource accounting. [Validation record](docs/VALIDATION.md) · [MIT license](LICENSE)

The published recording copies contain the same decoded video frames as the originals, with audio and GPS/device metadata removed. [Publication verification](data/raw/personal_circle/publication_verification.json) records both container hashes and identical decoded-frame hashes. The original files are preserved outside this repository.
