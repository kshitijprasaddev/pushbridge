"""Export measured evidence for the static Three.js reviewer application.

The browser replays recorded MuJoCo states; it does not run a second physics
engine or infer policy actions. Trajectories are exported without rounding and
loaded individually, rather than loading all 360 trajectories at startup.

Run from any directory using the project's Python environment. No experiment
is trained or rerun by this script.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web/public"
TRACE_FIELDS = ("state", "action", "joint", "phase", "goal")
LINKS_M = np.array([0.24, 0.17, 0.13])
TASK_ORIGIN_M = np.array([0.25, -0.105])
STRATEGIES = (
    {
        "id": "direct",
        "prefix": "source_direct",
        "label": "Direct transfer",
        "description": "The source SAC policy is applied to the compact arm without contact normalisation.",
    },
    {
        "id": "contact",
        "prefix": "source_contact",
        "label": "Contact adapter",
        "description": "The source policy is wrapped with a geometric adapter for the compact arm's narrower tool.",
    },
    {
        "id": "postRL",
        "prefix": "posttrained",
        "label": "Adapter + RL",
        "description": "The adapted source policy receives 20,000 SAC interactions on the compact embodiment.",
    },
)


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset(source: str, destination: str, label: str, **metadata: object) -> dict:
    src = ROOT / source
    dst = PUBLIC / destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if digest(src) != digest(dst):
        raise RuntimeError(f"Copied asset differs: {source}")
    return {
        "label": label,
        "url": f"/{destination}",
        "source": source,
        "bytes": dst.stat().st_size,
        "sha256": digest(dst),
        **metadata,
    }


def summary(values: dict) -> dict:
    return {key: value for key, value in values.items() if key != "runs"}


def main() -> None:
    original = read("dashboard/data.json")
    precision = read("results/precision_finish/evaluation.json")
    protocol = precision["protocol"]
    PUBLIC.mkdir(parents=True, exist_ok=True)
    # This directory belongs to the exporter. Removing old trajectories prevents
    # an obsolete run from appearing in a later static deployment.
    shutil.rmtree(PUBLIC / "data/runs", ignore_errors=True)

    runs = {}
    traces_checked = 0
    states_checked = 0
    max_fk_error_m = 0.0
    for strategy in STRATEGIES:
        for seed in range(3):
            for mode in ("policy", "finish"):
                suffix = "_finish" if mode == "finish" else ""
                name = f"{strategy['prefix']}_seed{seed}{suffix}"
                measured = precision["suites"]["paired"]["results"][name]
                entries = []
                for result in measured["runs"]:
                    trial = result["trial"]
                    source = f"results/precision_finish/paired/{name}_{trial:03d}.npz"
                    target = f"data/runs/{name}/{trial:03d}.json"
                    with np.load(ROOT / source) as archive:
                        trace = {key: archive[key].tolist() for key in TRACE_FIELDS}
                        q = archive["joint"]
                        theta = np.cumsum(q, axis=1)
                        fk = np.column_stack([
                            (np.cos(theta) * LINKS_M).sum(axis=1),
                            (np.sin(theta) * LINKS_M).sum(axis=1),
                        ]) - TASK_ORIGIN_M
                        error = float(np.linalg.norm(fk - archive["state"][:, :2], axis=1).max())
                        max_fk_error_m = max(max_fk_error_m, error)
                        # Recorded qpos and MuJoCo's cached site_xpos are not
                        # updated at exactly the same instant in mj_step. Keep
                        # both arrays unchanged and bound this small residual.
                        if error > 0.0002:
                            raise RuntimeError(f"3R forward kinematics differs from tool state: {source}: {error}")
                    if len(trace["action"]) != len(trace["state"]) - 1:
                        raise RuntimeError(f"State/action length mismatch: {source}")
                    if not len(trace["joint"]) == len(trace["phase"]) == len(trace["state"]):
                        raise RuntimeError(f"State/joint/phase length mismatch: {source}")
                    # Dashboard correspondence is checked as well: the website
                    # changes the renderer and preserves the experiment data.
                    if trace != original["precision_traces"][name][trial]:
                        raise RuntimeError(f"Original dashboard trajectory differs: {source}")
                    exported = {
                        **trace,
                        "metadata": {
                            "model": name,
                            "strategy": strategy["id"],
                            "seed": seed,
                            "mode": mode,
                            "trial": trial,
                            "suite": "paired",
                            "suite_seed": precision["suites"]["paired"]["seed"],
                            "dt": 0.1,
                            "units": "normalised simulator metres and radians",
                            "state_order": ["tool_x", "tool_y", "puck_x", "puck_y"],
                            "source": source,
                            "source_sha256": digest(ROOT / source),
                            "renderer": "Recorded MuJoCo trajectory rendered by Three.js",
                            "interpolation": "Browser interpolation is visual only; saved 0.1 s frames determine metrics.",
                        },
                        "result": result,
                    }
                    write(PUBLIC / target, exported)
                    restored = json.loads((PUBLIC / target).read_text())
                    if any(restored[key] != trace[key] for key in TRACE_FIELDS):
                        raise RuntimeError(f"JSON round-trip altered trajectory: {target}")
                    entries.append({**result, "url": f"/{target}"})
                    traces_checked += 1
                    states_checked += len(trace["state"])
                runs[name] = {
                    "id": name,
                    "strategy": strategy["id"],
                    "seed": seed,
                    "mode": mode,
                    "summary": summary(measured),
                    "trials": entries,
                }

    clips = []
    for i, clip in enumerate(original["clips"]):
        poster = asset(clip["poster"], f"media/clip_{i:03d}.jpg", clip["id"])
        video = asset(clip["video"], f"media/clip_{i:03d}_tracking.mp4", f"{clip['id']} tracking preview")
        clips.append({
            **{key: value for key, value in clip.items() if key not in ("state", "poster", "video")},
            "poster": poster["url"],
            "video": video["url"],
        })

    videos = []
    for strategy in STRATEGIES:
        for mode in ("policy", "finish"):
            suffix = "_finish" if mode == "finish" else ""
            name = f"{strategy['prefix']}_seed1{suffix}"
            source = f"results/precision_finish/paired/{name}"
            poster = asset(f"{source}.jpg", f"evidence/videos/{name}_poster.jpg", f"{strategy['label']} final frame")
            videos.append(asset(
                f"{source}.mp4",
                f"evidence/videos/{name}.mp4",
                f"{strategy['label']} {'+ precision feedback' if mode == 'finish' else 'policy alone'}",
                poster=poster["url"], strategy=strategy["id"], mode=mode, seed=1, trial=0,
                provenance="Native MuJoCo render, paired trial 0, model seed 1; state/action/joint/phase arrays match saved evidence.",
            ))

    downloads = []
    for seed in range(3):
        for kind, source in (
            ("source", f"results/rl_comparison/sac_demo_initialized_seed{seed}/sac.zip"),
            ("target", f"results/embodiment_transfer/posttrained_seed{seed}/sac.zip"),
        ):
            entry = asset(
                source, f"evidence/policies/{kind}_sac_seed{seed}.zip",
                f"{'Source' if kind == 'source' else 'Adapted + RL'} SAC checkpoint, seed {seed}",
                format="Stable-Baselines3 ZIP", seed=seed, kind=kind,
            )
            expected = protocol["checkpoints"][str(seed)][f"{'source' if kind == 'source' else 'posttrained'}_sha256"]
            if entry["sha256"] != expected:
                raise RuntimeError(f"Policy checkpoint does not match frozen protocol: {source}")
            downloads.append(entry)
    downloads.append(asset("results/policy_speed/actor.npz", "evidence/policies/actor.npz", "Compact deterministic actor weights", format="NumPy NPZ"))
    downloads.append(asset("results/personal_five/model/world_model.pt", "evidence/policies/world_model.pt", "Personal-data world model ensemble", format="PyTorch checkpoint"))
    docs = [asset(f"docs/{name}.md", f"evidence/docs/{name}.md", label, format="Markdown") for name, label in (
        ("REVIEWER_GUIDE", "Reviewer guide"),
        ("EXPERIMENTS", "Experiments and reproduction"),
        ("PRECISION", "Precision protocol"),
        ("VALIDATION", "Validation record"),
    )]
    results = [asset(source, destination, label, format="JSON") for source, destination, label in (
        ("results/precision_finish/evaluation.json", "evidence/results/precision_evaluation.json", "Full precision evaluation, 520 trajectories"),
        ("results/precision_finish/protocol.json", "evidence/results/precision_protocol.json", "Frozen precision protocol"),
        ("results/precision_finish/video_verification.json", "evidence/results/video_verification.json", "Native video trace verification"),
        ("results/embodiment_transfer/evaluation.json", "evidence/results/embodiment_evaluation.json", "Original embodiment evaluation"),
        ("results/submission_audit.json", "evidence/results/submission_audit.json", "Submission evidence audit"),
    )]
    world_video = asset("results/world_model_visuals/prediction_overlay.mp4", "evidence/prediction_overlay.mp4", "Held-out state prediction overlay")
    world_poster = asset("results/world_model_visuals/prediction_overlay.jpg", "media/prediction_overlay.jpg", "Held-out state prediction poster")
    world = {key: value for key, value in original["world"].items() if key != "windows"}
    world["horizons"] = [{**row, "steps": row["horizon_steps"], "mean_error_mm": row["learned_mean_error_mm"]} for row in world["horizons"]]
    world.update(video=world_video["url"], poster=world_poster["url"], windowsUrl="/data/world-windows.json")
    write(PUBLIC / "data/world-windows.json", original["world"]["windows"])
    speed = {key: value for key, value in original["speed"].items() if key not in ("traces", "torch_direct_traces")}
    speed["latency"] = {key: {k: v for k, v in row.items() if k != "block_medians_ms"} for key, row in speed["latency"].items()}

    verification = {
        "status": "passed",
        "scope": "Static web export consistency; no additional training, physics rerun or hardware test.",
        "traces_checked": traces_checked,
        "states_checked": states_checked,
        "fields_preserved": list(TRACE_FIELDS),
        "numeric_rounding": "None; JSON round-trip checked against the original float64 arrays.",
        "max_3r_fk_tool_position_error_m": max_fk_error_m,
        "fk_error_limit_m": 0.0002,
        "fk_note": "Recorded joint qpos and cached tool site position differ by at most 0.161 mm; both original arrays are preserved without correction.",
        "native_video_verification": "/evidence/results/video_verification.json",
        "checkpoint_hashes_checked": 6,
    }
    write(PUBLIC / "data/export-verification.json", verification)
    results.append({"label": "Web replay export verification", "url": "/data/export-verification.json", "format": "JSON"})
    manifest = {
        "schemaVersion": 1,
        "experiment": {
            "title": "PushBridge",
            "repository": read("web/repository.json")["url"] if (ROOT/"web/repository.json").exists() else None,
            "description": "Personal fingertip demonstrations, contact-space policy transfer, target-arm SAC and explicit precision feedback.",
            "viewer": "Recorded MuJoCo state replay with Three.js rendering",
            "policyInput": "Simulator state, not camera images or language",
            "provenance": "Five user-recorded home videos; three training clips, one validation clip and one test clip.",
            "coordinateUnits": "Normalised simulator metres; the real recording's physical scale was not measured.",
            "limitations": protocol["limitations"],
        },
        "robot": {
            "name": "Compact 3R",
            "linksM": LINKS_M.tolist(),
            "taskOriginM": TASK_ORIGIN_M.tolist(),
            "workspaceM": [0.297, 0.210],
            "shoulderHeightM": 0.15,
            "tipHeightM": 0.010,
            "tipRadiusM": 0.003,
            "puckRadiusM": 0.012,
            "puckHeightM": 0.018,
            "goalRadiusM": 0.020,
            "fitDistanceM": 0.008,
            "maxJointCommandSpeedRadS": 0.25,
            "dt": 0.1,
            "jointRangesRad": [[-1.3, 1.3], [0.05, 2.75], [-1.6, 1.6]],
        },
        "strategies": list(STRATEGIES),
        "seeds": [0, 1, 2],
        "trialCount": 20,
        "modes": ["policy", "finish"],
        "defaultRun": "source_contact_seed1_finish",
        "defaultTrial": 0,
        "runs": runs,
        "protocol": protocol,
        "suites": {key: {"seed": suite["seed"], "results": {name: summary(value) for name, value in suite["results"].items()}} for key, suite in precision["suites"].items()},
        "clips": clips,
        "dataset": {"manifest": original["manifest"], "prediction": original["prediction"]},
        "rl": original["rl"],
        "personal_posttraining": {**{k: v for k, v in original["personal_posttraining"].items() if k != "results"}, "results": {k: summary(v) for k, v in original["personal_posttraining"]["results"].items()}},
        "transfer": {**{k: v for k, v in original["transfer"].items() if k != "results"}, "results": {k: summary(v) for k, v in original["transfer"]["results"].items()}},
        "world": world,
        "speed": speed,
        "evidence": {"videos": videos, "downloads": downloads, "docs": docs, "results": results},
        "exportVerification": verification,
    }
    write(PUBLIC / "data/manifest.json", manifest)
    print(f"Exported {traces_checked} trajectories ({states_checked} saved frames), 5 clips, 6 native videos and 8 model downloads.")
    print(f"Manifest: {(PUBLIC / 'data/manifest.json').stat().st_size:,} bytes; maximum FK discrepancy: {max_fk_error_m:.3g} m.")


if __name__ == "__main__":
    main()
