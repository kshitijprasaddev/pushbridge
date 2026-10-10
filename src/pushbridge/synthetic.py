"""Explicitly synthetic fixtures; never presented as personally recorded data."""
from pathlib import Path

import cv2
import numpy as np

from .calibration import save_calibration
from .data import write_json
from .policy import geometric_action
from .sim import PushEnv
from .video import VideoWriter


def make_synthetic(out, episodes=10, seed=19):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob('*.mp4')):
        raise ValueError('Synthetic output folder already contains clips; use a new folder.')
    corners = [[60, 60], [654, 60], [654, 480], [60, 480]]
    frame = np.full((540, 720, 3), 34, np.uint8)
    cv2.rectangle(frame, (60, 60), (654, 480), (235, 235, 235), -1)
    save_calibration(frame, corners, out/'calibration.json')
    rng = np.random.default_rng(seed)
    for i in range(episodes):
        puck = rng.uniform([0.065, 0.065], [0.12, 0.15])
        goal = rng.uniform([0.21, 0.06], [0.255, 0.15])
        direction = (goal-puck)/np.linalg.norm(goal-puck)
        env = PushEnv(friction=float(rng.uniform(0.15, 0.4)))
        state = env.reset(puck, puck-direction*0.04, goal)
        writer = VideoWriter(out/f'synthetic_{i:02d}.mp4', 10, (720, 540))
        if not writer.isOpened():
            raise RuntimeError('Video encoder unavailable')
        truth = []
        try:
            for k in range(80):
                image = frame.copy()
                def pixel(xy):
                    return tuple(np.round([60+xy[0]*2000, 480-xy[1]*2000]).astype(int))
                cv2.circle(image, pixel(goal), 34, (90, 165, 80), 2)
                cv2.circle(image, pixel(state[2:]), 24, (35, 35, 230), -1)
                cv2.circle(image, pixel(state[:2]), 13, (230, 90, 20), -1)
                cv2.putText(image, 'SYNTHETIC SIMULATOR MARKERS - NO HUMAN RECORDING', (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 220, 245), 1)
                writer.write(image)
                truth.append(state.copy())
                action = geometric_action(state, goal) if np.linalg.norm(state[2:]-goal) > 0.008 else np.zeros(2)
                state = env.step(action)
        finally:
            writer.release()
            env.close()
        np.savez_compressed(out/f'synthetic_{i:02d}_truth.npz', state=np.array(truth))
        print(f'Synthetic clip {i+1}/{episodes}: final error {1000*np.linalg.norm(state[2:]-goal):.1f} mm', flush=True)
    write_json(out/'SYNTHETIC_ONLY.json', {'source': 'synthetic', 'seed': seed, 'episodes': episodes,
                                        'warning': 'Not personally collected data; do not submit these as real recordings.'})
