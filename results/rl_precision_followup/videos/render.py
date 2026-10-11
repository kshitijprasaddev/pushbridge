"""Render a declared illustrative case and verify exact recorded physics.

From the repository root:
python results/rl_precision_followup/videos/render.py

This is the old seed2 Task04 failure, not a best-performing new case.
All 760 follow-up outcomes remain available independently of these videos.
"""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from precision_experiment import render_observer
from pushbridge.data import sha256, write_json
from pushbridge.embodiment import ContactAdapter
from pushbridge.evaluation import make_trials
from pushbridge.precision import PrecisionHandoff, run_precision
from pushbridge.rl import SACPolicy
from pushbridge.video import VideoWriter


OUT = ROOT / 'results/rl_precision_followup'
SPECS = [
    {'run': 'original_20k_seed2', 'title': 'Original RL, 20k decisions',
     'banner': 'Original target SAC | 20k decisions | seed 2 / task 04',
     'checkpoint': 'results/embodiment_transfer/posttrained_seed2/sac.zip',
     'description': 'The original policy misses the target on this previously recorded failure. Exact MuJoCo state is supplied to the actor.'},
    {'run': 'longer_original_seed2', 'title': 'Longer RL, 60k decisions',
     'banner': 'Longer target SAC | 60k decisions | seed 2 / task 04',
     'checkpoint': 'results/rl_precision_followup/longer_original_seed2/sac.zip',
     'description': 'Another 40k decisions with the same 18 mm training objective brings the puck closer. This case still fails the stricter 8 mm containment rule.'},
    {'run': 'longer_original_seed2_finish', 'title': 'Longer RL with the finisher',
     'banner': '60k target SAC + feedback | seed 2 / task 04',
     'checkpoint': 'results/rl_precision_followup/longer_original_seed2/sac.zip',
     'description': 'The same 60k policy handles the approach. The unchanged hand-coded finisher takes over, releases contact and passes the one-second containment check.'},
]


def main():
    torch.set_num_threads(2)
    trial = make_trials(20, 2027)[3]
    videos = []
    for spec in SPECS:
        name = spec['run']
        policy = ContactAdapter(SACPolicy(ROOT / spec['checkpoint']))
        controller = PrecisionHandoff(policy) if name.endswith('_finish') else policy
        movie = OUT / 'videos' / f'{name}.mp4'
        poster = OUT / 'videos' / f'{name}.jpg'
        writer = VideoWriter(movie, 10, (960, 640))
        observer, last_frame = render_observer(writer, spec['banner'], trial)
        saved_path = OUT / 'paired' / f'{name}_003.npz'
        try:
            row, trace = run_precision(controller, trial, observer=observer)
            with np.load(saved_path, allow_pickle=False) as saved:
                for key in ('state', 'action', 'joint', 'phase', 'goal'):
                    np.testing.assert_array_equal(trace[key], saved[key])
            frame = last_frame[0]
            status = 'FULL OBJECT INSIDE | 1 s HOLD PASSED' if row['success'] else 'INCOMPLETE | OBJECT DOES NOT FIT'
            cv2.rectangle(frame, (16, 588), (650, 624), (31, 41, 38), -1)
            cv2.putText(frame, status, (28, 611), cv2.FONT_HERSHEY_SIMPLEX, .51,
                        (147, 226, 192) if row['success'] else (137, 186, 235), 1, cv2.LINE_AA)
            for _ in range(20):
                writer.write(frame)
            if not cv2.imwrite(str(poster), frame):
                raise RuntimeError(f'Cannot save poster: {poster}')
        finally:
            writer.release()
        videos.append({
            'id': name, 'run': name, 'title': spec['title'], 'description': spec['description'],
            'file': str(movie.relative_to(ROOT)), 'poster': str(poster.relative_to(ROOT)),
            'file_sha256': sha256(movie), 'poster_sha256': sha256(poster),
            'source': {'checkpoint': spec['checkpoint'], 'checkpoint_sha256': sha256(ROOT / spec['checkpoint']),
                       'trace': str(saved_path.relative_to(ROOT)), 'trace_sha256': sha256(saved_path)},
            'suite': 'paired', 'evaluation_seed': 2027, 'model_seed': 2, 'trial': 3, 'task': 4,
            'success': row['success'], 'final_distance_mm': row['final_distance_mm'],
            'simulation_duration_s': len(trace['action']) * .1, 'final_image_display_seconds': 2,
            'frame_rate': 10, 'resolution': [960, 640],
            'verification': {'all_states_actions_joints_and_phases_bit_identical': True,
                             'max_state_difference': 0.},
        })
        print(f'{name}: verified exact physics, error {row["final_distance_mm"]:.3f} mm', flush=True)
    write_json(OUT / 'videos.json', {
        'selection': 'Illustrative paired seed2 Task04, chosen because it was a previously visible original failure; no new model or checkpoint was selected using this case.',
        'scope': 'Native MuJoCo rendering of physically re-simulated saved runs. No animation scripting, object attachment or changed physics.',
        'suite': 'paired', 'evaluation_seed': 2027, 'model_seed': 2, 'trial': 3, 'task': 4,
        'all_followup_trials_retained': 760, 'videos': videos,
    })


if __name__ == '__main__':
    main()
