# The 3D research app

The main presentation is a React app built with Vite. Three.js reconstructs the compact three-joint robot and its pushes from saved MuJoCo experiments. Reviewers can compare the same task across strategies, inspect the contact and replay a recorded failure. The production build is a static site that runs in a browser.

## Run locally

Use Node 24.x and pnpm 11.25.0, matching the local verification and CI. The repository pins the package-manager version in `web/package.json` and dependencies in `web/pnpm-lock.yaml`. These commands use the pinned pnpm version without requiring a separate installation:

```bash
cd web
npx --yes pnpm@11.25.0 install --frozen-lockfile
npx --yes pnpm@11.25.0 dev
```

Open the URL printed by Vite. To inspect the production build:

```bash
npx --yes pnpm@11.25.0 run build
npx --yes pnpm@11.25.0 preview --host 127.0.0.1
```

If pnpm 11.25.0 is already installed, use `pnpm` directly. The repository-root shortcuts are `make web-install`, `make web-dev`, `make web-build` and `make web-preview`. The frontend build packages the included evidence; it does not need Python or MuJoCo, or run training.

## What the replay means

The robot's joint angles, tool and object positions, target, control phases and outcomes come from saved physics traces. Scrubbing selects a recorded time. Smooth motion between samples is visual interpolation, while MuJoCo generated the actual contacts. Experiments use a 0.1-second control interval with smaller internal physics steps.

MuJoCo records joint positions and a cached tool-site position at slightly different points in its integration step. Their forward-kinematics difference is at most 0.161 mm across the exported traces. Both arrays are preserved. The 3D arm follows the recorded joints, while the tool-state readout uses the recorded site position; neither is adjusted to hide that residual.

The target radius is 20 mm and the object radius is 12 mm. Full containment therefore requires a centre distance no greater than 8 mm. A successful strict trial also keeps the object contained during ten zero-action control steps. Rendering quality does not alter this success criterion.

Use **Policy only** to inspect what the learned policy accomplishes on its own. **+ Precision finish** adds the hand-coded controller when the object reaches 30 mm from the goal. It aligns the tool, pushes, releases contact and verifies the hold. The results table reports the two modes separately, and recorded failures remain available for every training seed.

The strategy, model-seed and task selectors choose one of 360 recorded trajectories. **Download this trace** matches that selection. The original MuJoCo video is explicitly labelled as reference task 01, seed 1; it is not available for every selected trajectory.

The trained actors use exact simulator positions in a planar manipulation task. The website replays these experiments and does not connect to robot hardware or execute a new physics rollout. Image- and language-conditioned policies are outside this study.

## Refresh the public evidence

The app's public files are exported from the reference experiment artifacts. `web/public/data/manifest.json` holds the overview; individual replays are fetched from `web/public/data/runs/` as needed. Selected videos, documents, results and policy files live under `web/public/media/` and `web/public/evidence/`.

After installing the Python project dependencies, run from the repository root:

```bash
python scripts/export_web_data.py
python scripts/audit_submission.py
```

Or use `make web-data` and `make audit`. The exporter prepares browser data and selected assets; it does not retrain policies. Its `data/export-verification.json` compares exported states, actions, joint angles, phases and goals against the saved traces, and checks forward kinematics against the tool positions. Keep the exported public files in Git so a clean Vercel build does not need the Python environment. Policy checkpoints, protocols and full traces remain in `results/` and `data/`; publication copies of the recordings are in `data/raw/personal_circle/`.

To reproduce the physics results, follow [the experiment commands](EXPERIMENTS.md) and [the precision protocol](PRECISION.md). The frontend build and the experiment rerun are separate operations.

## Deploy to Vercel

Push the complete repository to GitHub, then import it into Vercel with these project settings:

| Setting          | Value                                              |
| ---------------- | -------------------------------------------------- |
| Root directory   | `web`                                              |
| Framework preset | Vite                                               |
| Install command  | `npx --yes pnpm@11.25.0 install --frozen-lockfile` |
| Build command    | `npx --yes pnpm@11.25.0 run build`                 |
| Output directory | `dist`                                             |
| Node version     | `24.x`                                             |

The settings are also recorded in `web/vercel.json`, with Node 24.x in `web/package.json`. Explicitly invoking pnpm 11.25.0 keeps Vercel from choosing a different major version from the lockfile format. No environment variables or Corepack setting are required.

Commit the lockfile and exported public evidence. The app needs no Python server, GPU backend or training service on Vercel. Keep `node_modules/`, local environment files, `.vercel/` and build caches out of Git.

Review the production preview before promoting it and use the resulting HTTPS address in the submission. These instructions prepare the deployment; no public deployment is implied by the local demo.

## Repository responsibilities

| Location                                 | Purpose                                               |
| ---------------------------------------- | ----------------------------------------------------- |
| `web/src/`                               | React interface and Three.js renderer                 |
| `web/public/`                            | Exported browser evidence and presentation assets     |
| `web/package.json`, `web/pnpm-lock.yaml` | Frontend scripts and exact dependency resolution      |
| `scripts/export_web_data.py`             | Regenerate public evidence from reference results     |
| `src/pushbridge/`                        | Tracking, learning, policies, controllers and physics |
| `results/`                               | Original measurements, checkpoints, traces and videos |
| `index.html`, `dashboard/`               | Retained earlier offline evidence report              |

The submission ZIP contains frontend source and public evidence, with dependencies and generated `dist/` output excluded. Run `make package` after the Python audit and frontend build checks.

GitHub CI runs the Python package checks and evidence audit in one job, and installs the locked frontend dependencies, runs its numerical replay checks and builds the production app in another. A passing local build does not imply a completed GitHub run or public deployment.

Before public GitHub publication, the phone recordings were remuxed without audio or location/device metadata. The compressed video packets were copied, and full decoded-frame hashes, frame counts, rates and sizes match the originals. See `data/raw/personal_circle/publication_verification.json`. The originals remain outside the repository.
