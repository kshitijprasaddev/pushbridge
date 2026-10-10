"""Refresh six presentation videos while checking exact saved-state replay."""
from pathlib import Path

import cv2
import numpy as np
import torch

from precision_experiment import render_observer, video_title
from pushbridge.data import write_json
from pushbridge.embodiment import ContactAdapter
from pushbridge.evaluation import make_trials
from pushbridge.precision import PrecisionHandoff, run_precision
from pushbridge.rl import SACPolicy
from pushbridge.video import VideoWriter


ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    out = ROOT/'results/precision_finish/paired'
    source = SACPolicy(ROOT/'results/rl_comparison/sac_demo_initialized_seed1/sac.zip')
    post = SACPolicy(ROOT/'results/embodiment_transfer/posttrained_seed1/sac.zip')
    policies = {'source_direct_seed1': source, 'source_contact_seed1': ContactAdapter(source),
                'posttrained_seed1': ContactAdapter(post)}
    trial, checks = make_trials(1, 2027)[0], {}
    for base, policy in policies.items():
        for hybrid in [False, True]:
            name = base + ('_finish' if hybrid else '')
            controller = PrecisionHandoff(policy) if hybrid else policy
            writer = VideoWriter(out/f'{name}.mp4', 10, (960, 640))
            observer, last_frame = render_observer(writer, video_title(name), trial)
            try:
                row, trace = run_precision(controller, trial, observer=observer)
                with np.load(out/f'{name}_000.npz') as saved:
                    for key in ['state', 'action', 'joint', 'phase', 'goal']:
                        np.testing.assert_array_equal(trace[key], saved[key])
                frame = last_frame[0]
                status = 'FULL OBJECT INSIDE | 1 s HOLD PASSED' if row['success'] else 'INCOMPLETE | OBJECT DOES NOT FIT'
                cv2.rectangle(frame, (16, 588), (630, 624), (31, 41, 38), -1)
                cv2.putText(frame, status, (28, 611), cv2.FONT_HERSHEY_SIMPLEX, .51,
                            (147, 226, 192) if row['success'] else (137, 186, 235), 1, cv2.LINE_AA)
                for _ in range(20):
                    writer.write(frame)
                cv2.imwrite(str(out/f'{name}.jpg'), frame)
            finally:
                writer.release()
            checks[name] = {'all_states_actions_joints_and_phases_bit_identical': True,
                            'max_state_difference': 0., 'success': row['success'],
                            'simulation_duration_s': len(trace['action'])*.1, 'final_image_display_seconds': 2}
            print(f'{name}: exact physics replay, {row["final_distance_mm"]:.1f} mm', flush=True)
    write_json(ROOT/'results/precision_finish/video_verification.json', checks)


if __name__ == '__main__':
    main()
