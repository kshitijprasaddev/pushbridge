import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episodes(folder):
    manifest = json.loads((Path(folder)/"manifest.json").read_text())
    episodes = []
    for item in manifest["episodes"]:
        with np.load(Path(folder)/item["file"], allow_pickle=False) as d:
            episodes.append({**item, "state": d["state"], "time": d["time"], "valid": d["valid"].astype(bool)})
            if 'video_time_s' in d:
                episodes[-1]['video_time_s'] = d['video_time_s'].copy()
    return manifest, episodes


def transitions(episode):
    s, t, valid = episode["state"], episode["time"], episode["valid"]
    # Never bridge occlusions or dropped detections. One action is 0.1 seconds.
    good = valid[:-1] & valid[1:] & np.isclose(np.diff(t), 0.1, atol=1e-5)
    good &= np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1) <= 0.025
    good &= np.linalg.norm(np.diff(s[:, 2:], axis=0), axis=1) <= 0.025
    indices = np.flatnonzero(good)
    return s[indices], s[indices+1, :2]-s[indices, :2], s[indices+1, 2:]-s[indices, 2:], indices


def episode_split(episodes, seed=7):
    if len(episodes) < 5:
        raise ValueError("Need at least 5 different clips; 10 is recommended for a useful whole-clip split.")
    order = np.random.default_rng(seed).permutation(len(episodes))
    hold = max(1, len(episodes)//5)
    return ([episodes[i] for i in order[2*hold:]],
            [episodes[i] for i in order[:hold]], [episodes[i] for i in order[hold:2*hold]])
