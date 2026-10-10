"""Replay a stored failed trial through physics and verify its saved trajectory."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pushbridge.data import write_json
from pushbridge.sim import PushEnv
from pushbridge.video import VideoWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--trial', type=int)
    args = parser.parse_args()
    out = args.evaluation
    report = json.loads((out/'simulation_metrics.json').read_text())
    runs = report['results'][args.mode]['runs']
    trial = next(r for r in runs if r['trial'] == args.trial) if args.trial is not None else next(r for r in runs if not r['success'])
    filename = f'{args.mode}_failure_{trial["trial"]:03d}'
    trajectory = np.load(out/f'{args.mode}_{trial["trial"]:03d}.npz', allow_pickle=False)
    env = PushEnv(friction=trial['friction'])
    env.reset(trial['initial_puck'], trial['initial_tool'], trial['goal'])
    writer = VideoWriter(out/f'{filename}.mp4', 10, (960, 640))
    errors = []
    try:
        for i, action in enumerate(trajectory['action']):
            state = env.step(action)
            errors.append(np.linalg.norm(state-trajectory['state'][i+1]))
            frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (960, 82), (24, 29, 39), -1)
            distance = np.linalg.norm(state[2:]-trial['goal'])*1000
            cv2.putText(frame, f'{args.mode} | trial {trial["trial"]} | {distance:.1f} mm', (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(frame, 'Stored trial actions replayed through MuJoCo contact', (18, 64), cv2.FONT_HERSHEY_SIMPLEX, .50, (100, 210, 245), 1)
            writer.write(frame)
            if i == len(trajectory['action'])-1:
                cv2.imwrite(str(out/f'{filename}.jpg'), frame)
    finally:
        writer.release()
        env.close()
    if max(errors) > 1e-7:
        raise ValueError('Physics replay differs from the saved trajectory')
    write_json(out/f'{filename}.json', {'trial': trial, 'selection': 'First failed trial unless explicitly selected',
               'frames': len(errors), 'max_saved_state_replay_difference_mm': float(max(errors)*1000),
               'last_20_requested_action_mean_norm_mm': float(np.linalg.norm(trajectory['action'][-20:], axis=1).mean()*1000)})
    print(f'{filename}: replay verified, maximum state error {max(errors)*1000:.3g} mm')


if __name__ == '__main__':
    main()
