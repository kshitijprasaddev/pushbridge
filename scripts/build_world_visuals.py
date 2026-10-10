"""Held-out open-loop state predictions projected back onto real footage."""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pushbridge.data import load_episodes, transitions, write_json
from pushbridge.model import WorldModel, analytic_delta
from pushbridge.video import VideoWriter


ROOT = Path(__file__).resolve().parents[1]


def pixel_position(position, extraction, index):
    # Invert the same per-frame task transform used to ingest the recording.
    relative = (np.asarray(position)-[.225, .105])@extraction['task_frame'].T
    relative = relative*extraction['target_radius_px'][index]/.020
    relative[1] *= -1
    return relative+extraction['target_px'][index]


def main():
    torch.set_num_threads(2)
    dataset = ROOT/'data/processed/personal_five'
    out = ROOT/'results/world_model_visuals'
    out.mkdir(parents=True, exist_ok=True)
    world = WorldModel.load(ROOT/'results/personal_five/model/world_model.pt')
    manifest, episodes = load_episodes(dataset)
    test = [e for e in episodes if e['id'] in world.checkpoint['test_ids']]
    windows, metrics = [], []
    for horizon in [1, 5, 10, 20, 40]:
        errors = {'learned': [], 'zero': [], 'analytic': []}
        for ep in test:
            states = ep['state']
            valid = np.zeros(len(states)-1, bool)
            valid[transitions(ep)[3]] = True
            for start in range(0, len(states)-horizon, horizon):
                if not valid[start:start+horizon].all():
                    continue
                predicted, fixed = [states[start, 2:].copy()], [states[start, 2:].copy()]
                uncertainty = [0.]
                for k in range(start, start+horizon):
                    a = states[k+1, :2]-states[k, :2]
                    d, r = world.predict(states[k, :2]-predicted[-1], a)
                    predicted.append(predicted[-1]+d[0])
                    fixed.append(fixed[-1]+analytic_delta((states[k, :2]-fixed[-1])[None], a[None])[0])
                    uncertainty.append(uncertainty[-1]+float(r[0]))
                observed = states[start:start+horizon+1, 2:]
                for name, endpoint in [('learned', predicted[-1]), ('zero', observed[0]), ('analytic', fixed[-1])]:
                    errors[name].append(float(np.linalg.norm(endpoint-observed[-1])*1000))
                if horizon == 10:
                    windows.append({'clip_id': ep['id'], 'start_index': start,
                                    'video_time_s': ep.get('video_time_s', ep['time'])[start:start+horizon+1].tolist(),
                                    'observed': observed.tolist(), 'tool': states[start:start+horizon+1, :2].tolist(),
                                    'predicted': np.array(predicted).tolist(), 'analytic': np.array(fixed).tolist(),
                                    'heuristic_sum_one_step_radius_m': uncertainty,
                                    'endpoint_error_mm': errors['learned'][-1]})
        row = {'horizon_steps': horizon, 'horizon_seconds': horizon*.1, 'windows': len(errors['learned'])}
        row.update({name+'_mean_error_mm': float(np.mean(v)) for name, v in errors.items()})
        metrics.append(row)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    for name, label, color in [('learned', 'Personal-data world model', '#168a79'), ('analytic', 'Fixed contact dynamics', '#b98448'), ('zero', 'No object motion', '#94a2af')]:
        ax.plot([m['horizon_seconds'] for m in metrics], [m[name+'_mean_error_mm'] for m in metrics], '-o', label=label, color=color)
    ax.set(xlabel='Open-loop prediction horizon (seconds)', ylabel='Mean endpoint error (normalised mm)', title='Held-out recording: how quickly predictions drift')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out/'horizon_errors.svg')
    fig.savefig(out/'horizon_errors.png', dpi=180)
    plt.close(fig)
    # A video inspection artifact, not a neural video-generation claim.
    episode = test[0]
    index = next(i for i, e in enumerate(episodes) if e['id'] == episode['id'])
    extraction = np.load(dataset/f'{index:03d}_diagnostics/extracted_task.npz')
    video = ROOT/'data/raw/personal_circle'/episode['raw_name']
    cap = cv2.VideoCapture(str(video))
    writer = VideoWriter(out/'prediction_overlay.mp4', 10, (1120, 640))
    colors = {'observed': (75, 205, 90), 'predicted': (225, 190, 35), 'analytic': (60, 140, 230)}
    try:
        for window in windows:
            for j in range(11):
                k = window['start_index']+j
                cap.set(cv2.CAP_PROP_POS_MSEC, window['video_time_s'][j]*1000)
                ok, raw = cap.read()
                if not ok:
                    raise RuntimeError('Cannot read held-out video frame')
                for name in ['observed', 'predicted']:
                    xy = tuple(np.round(pixel_position(window[name][j], extraction, k)).astype(int))
                    cv2.drawMarker(raw, xy, colors[name], cv2.MARKER_CROSS if name == 'predicted' else cv2.MARKER_DIAMOND, 32, 4)
                left = source_frame_from_image(raw)
                cv2.rectangle(left, (0, 0), (480, 75), (20, 27, 35), -1)
                cv2.putText(left, 'HELD-OUT PERSONAL RECORDING', (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .47, (235, 240, 240), 1)
                cv2.putText(left, 'Green: tracked | cyan: predicted', (12, 53), cv2.FONT_HERSHEY_SIMPLEX, .45, (110, 220, 200), 1)
                right = np.full((640, 640, 3), (245, 247, 246), np.uint8)
                def xy(p):
                    return tuple(np.round(np.array([55+530*p[0]/.297, 555-405*p[1]/.210])).astype(int))
                cv2.rectangle(right, (55, 150), (585, 555), (212, 220, 217), 1)
                cv2.circle(right, xy([.225, .105]), round(530*.020/.297), (180, 215, 185), 1)
                for name in ['analytic', 'predicted', 'observed']:
                    path = np.array([xy(p) for p in window[name][:j+1]], np.int32)
                    cv2.polylines(right, [path], False, colors[name], 3)
                    cv2.circle(right, tuple(path[-1]), 7, colors[name], -1)
                tool = xy(window['tool'][j])
                cv2.circle(right, tool, 5, (145, 135, 130), -1)
                error = np.linalg.norm(np.array(window['predicted'][j])-window['observed'][j])*1000
                for y, text in [(35, 'OPEN-LOOP STATE PREDICTION'), (65, f'{j} / 10 steps | endpoint error {error:.2f} normalised mm'),
                                (95, 'Known future finger actions; object state uncorrected'), (605, 'Green: tracked | cyan: learned | orange: fixed contact')]:
                    cv2.putText(right, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, .48 if y != 35 else .65, (50, 65, 65), 1 if y != 35 else 2)
                frame = np.hstack([left, right])
                writer.write(frame)
                if window['start_index'] == 0 and j == 5:
                    cv2.imwrite(str(out/'prediction_overlay.jpg'), frame)
    finally:
        writer.release()
        cap.release()
        extraction.close()
    domains = {}
    paths = {'source_robot': ROOT/'results/rl_comparison/evaluation', 'compact_robot': ROOT/'results/embodiment_transfer'}
    patterns = {'source_robot': 'sac_demo_initialized_seed0_*.npz', 'compact_robot': 'posttrained_seed0_*.npz'}
    for domain, folder in paths.items():
        state, action, delta = [], [], []
        for p in sorted(folder.glob(patterns[domain])):
            with np.load(p) as d:
                s = d['state']
                state.extend(s[:-1])
                action.extend(np.diff(s[:, :2], axis=0))
                delta.extend(np.diff(s[:, 2:], axis=0))
        if not state:
            continue
        s, a, d = np.array(state), np.array(action), np.array(delta)
        pred, radius = world.predict(s[:, :2]-s[:, 2:], a)
        error = np.linalg.norm(pred-d, axis=1)
        moving = np.linalg.norm(d, axis=1) > .001
        domains[domain] = {'transitions': len(s), 'moving_transitions': int(moving.sum()),
                           'mean_error_mm': float(error.mean()*1000), 'moving_mean_error_mm': float(error[moving].mean()*1000),
                           'one_step_empirical_radius_coverage': float(np.mean(error <= radius)),
                           'input_action': 'Achieved tool displacement, not requested action',
                           'adaptation': 'None; evaluates the unmodified personal-data model across domains'}
    write_json(out/'metrics.json', {'test_ids': world.checkpoint['test_ids'], 'horizons': metrics, 'robot_domain_checks': domains,
                                  'windows': windows, 'type': 'Action-conditioned state prediction projected onto real recorded frames',
                                  'coordinate_units': 'Circle-normalised simulator metres; not measured real-world metres',
                                  'limitations': ['One held-out clip, no broad video-prediction generalisation claim',
                                                 'Future finger/tool motion is supplied, not predicted',
                                                 'Rendered overlays visualise predicted states; this is not a generative video model',
                                                 'One-step radius calibrated on validation; its accumulated sum is only a visual heuristic',
                                                 'Robot-domain errors are diagnostics, not additional model training']})
    print('World-model visualisation: held-out windows', len(windows), 'horizons', metrics, flush=True)


def source_frame_from_image(image):
    factor = min(480/image.shape[1], 640/image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*factor), round(image.shape[0]*factor)))
    canvas = np.zeros((640, 480, 3), np.uint8)
    x, y = (480-resized.shape[1])//2, (640-resized.shape[0])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


if __name__ == '__main__':
    main()
