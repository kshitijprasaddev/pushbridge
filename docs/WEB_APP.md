# The 3D research app

The main presentation is a React app built with Vite. Three.js displays the compact three-joint robot, target and object from saved MuJoCo experiments. A production build is a static site: reviewers need a browser, while Node is needed only to develop or build it.

## Run locally

Use Node 22.12 or newer. The repository records the package-manager version and exact frontend dependencies in `web/package.json` and `web/pnpm-lock.yaml`. Install pnpm using Corepack if it is available with your Node installation:

```bash
corepack enable
cd web
pnpm install --frozen-lockfile
pnpm dev
```

Open the URL printed by Vite. To inspect the production build:

```bash
pnpm build
pnpm preview --host 127.0.0.1
```

From the repository root, the same operations are available as `make web-install`, `make web-dev`, `make web-build` and `make web-preview`. Building the presentation uses the included public evidence and does not install MuJoCo, train a model or change experiment results.

## What the replay means

The visual scene is a renderer of recorded evidence. The robot's joint angles, tool and object positions, target, control phases and outcomes come from saved physics traces. Scrubbing selects a recorded time; smooth animation between samples is only visual interpolation. The MuJoCo experiment runs at a 0.1-second control interval with smaller internal physics steps.

MuJoCo records joint positions and a cached tool-site position at slightly different points in its integration step. Their forward-kinematics difference is at most 0.161 mm across the exported traces. Both arrays are preserved. The 3D arm follows the recorded joints, while the tool-state readout uses the recorded site position; neither is adjusted to hide that residual.

The target radius is 20 mm and the object radius is 12 mm. Full containment therefore requires a centre distance no greater than 8 mm. A successful strict trial also keeps the object contained during ten zero-action control steps. Rendering quality does not alter this success criterion.

The app distinguishes the frozen policy, the contact-radius adapter and explicit precision feedback. The hand-coded finisher handles the last 30 mm in hybrid runs. Its contribution must be attributed to feedback, rather than to a learned policy alone. Recorded failures and all training seeds remain available in the evidence.

This is a state-based planar manipulation study. The trained actors consume simulator positions. The browser scene does not claim image or language input, a live robot connection, a humanoid deployment, or a newly executed physics rollout.

## Refresh the public evidence

The app's public files are exported from the reference experiment artifacts. `web/public/data/manifest.json` holds the overview; individual replays are fetched from `web/public/data/runs/` as needed. Selected videos, documents, results and policy files live under `web/public/media/` and `web/public/evidence/`.

After installing the Python project dependencies, run from the repository root:

```bash
python scripts/export_web_data.py
python scripts/audit_submission.py
```

Or use `make web-data` and `make audit`. The exporter prepares browser data and selected assets; it does not retrain policies. Its `data/export-verification.json` compares exported states, actions, joint angles, phases and goals against the saved traces, and checks forward kinematics against the tool positions. Keep the exported public files in Git so a clean Vercel build does not need the Python environment. Policy checkpoints, protocols, full traces and original videos remain in `results/` and `data/`.

To reproduce the physics results, follow [the experiment commands](EXPERIMENTS.md) and [the precision protocol](PRECISION.md). The frontend build and the experiment rerun are separate operations.

## Deploy to Vercel

Push the complete repository to GitHub, then import it into Vercel with these project settings:

| Setting | Value |
| --- | --- |
| Root directory | `web` |
| Framework preset | Vite |
| Install command | `pnpm install --frozen-lockfile` |
| Build command | `pnpm build` |
| Output directory | `dist` |
| Node version | A supported version satisfying `>=22.12` |

The lockfile and exported public files must be committed. The app needs no Python server, GPU backend or training service on Vercel. Do not upload `node_modules/`, local environment files, `.vercel/`, or build caches.

Review the production preview before promoting it and use the resulting HTTPS address in the submission. These instructions prepare the deployment; no public deployment is implied by the local demo.

## Repository responsibilities

| Location | Purpose |
| --- | --- |
| `web/src/` | React interface and Three.js renderer |
| `web/public/` | Exported browser evidence and presentation assets |
| `web/package.json`, `web/pnpm-lock.yaml` | Frontend scripts and exact dependency resolution |
| `scripts/export_web_data.py` | Regenerate public evidence from reference results |
| `src/pushbridge/` | Tracking, learning, policies, controllers and physics |
| `results/` | Original measurements, checkpoints, traces and videos |
| `index.html`, `dashboard/` | Retained earlier offline evidence report |

The submission ZIP contains frontend source and public evidence, with dependencies and generated `dist/` output excluded. Run `make package` after the Python audit and frontend build checks.

GitHub CI runs the Python package checks and evidence audit in one job, and installs the locked frontend dependencies, runs its numerical replay checks and builds the production app in another. A passing local build does not imply a completed GitHub run or public deployment.

Before public GitHub publication, the phone recordings were remuxed without audio or location/device metadata. The compressed video packets were copied, and full decoded-frame hashes, frame counts, rates and sizes match the originals. See `data/raw/personal_circle/publication_verification.json`. The originals remain outside the repository.
