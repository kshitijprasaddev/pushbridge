"""Build the offline dashboard's data from measured experiment artifacts."""
from pathlib import Path
import json

import cv2
import numpy as np

from pushbridge.data import load_episodes, write_json


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    out = ROOT/'dashboard'
    out.mkdir(exist_ok=True)
    manifest, episodes = load_episodes(ROOT/'data/processed/personal_five')
    prediction = read('results/personal_five/model/prediction_metrics.json')
    roles = {identity: role for role, ids in prediction['split'].items() for identity in ids}
    clips = []
    for i, episode in enumerate(episodes):
        cap = cv2.VideoCapture(str(ROOT/'data/raw/personal_circle'/episode['raw_name']))
        cap.set(cv2.CAP_PROP_POS_MSEC, (episode['tracking']['continuous_segment_s'][0]+.5)*1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError('Cannot create source-recording poster')
        frame = cv2.resize(frame, (360, round(frame.shape[0]*360/frame.shape[1])))
        cv2.imwrite(str(out/f'clip_{i:03d}.jpg'), frame)
        clips.append({'id': episode['id'], 'role': roles[episode['id']].replace('_ids', ''),
                      'source': episode['raw_name'], 'transitions': episode['transitions'],
                      'moving_transitions': episode['moving_transitions'],
                      'retained_s': episode['tracking']['continuous_segment_s'],
                      'video': f'data/processed/personal_five/{i:03d}_diagnostics/tracking_preview.mp4',
                      'poster': f'dashboard/clip_{i:03d}.jpg', 'state': episode['state'].tolist(),
                      'raw_sha256': episode['raw_sha256']})
    transfer = read('results/embodiment_transfer/evaluation.json')
    speed = read('results/policy_speed/metrics.json')
    world = read('results/world_model_visuals/metrics.json')
    rl = read('results/rl_comparison/evaluation/simulation_metrics.json')
    personal_post = read('results/personal_posttraining/evaluation/simulation_metrics.json')
    precision = read('results/precision_finish/evaluation.json')
    precision_traces = {}
    for seed in range(3):
        for strategy in ['source_direct', 'source_contact', 'posttrained']:
            for suffix in ['', '_finish']:
                name = f'{strategy}_seed{seed}{suffix}'
                precision_traces[name] = []
                for trial in range(20):
                    with np.load(ROOT/f'results/precision_finish/paired/{name}_{trial:03d}.npz') as trace:
                        precision_traces[name].append({key: trace[key].tolist() for key in ['state', 'action', 'joint', 'phase', 'goal']})
    data = {'clips': clips, 'manifest': manifest, 'prediction': prediction,
            'rl': {k: {key: value for key, value in v.items() if key != 'runs'} for k, v in rl['results'].items()},
            'personal_posttraining': personal_post, 'transfer': transfer, 'world': world, 'speed': speed,
            'precision': precision, 'precision_traces': precision_traces}
    write_json(out/'data.json', data)
    # Embedding the exact same JSON permits opening index.html directly from
    # the submission ZIP, without browser file:// fetch restrictions.
    (out/'data.js').write_text('window.PUSHBRIDGE_DATA = '+json.dumps(data, allow_nan=False)+';\n')
    print('Dashboard data built from recorded metrics; five original clip posters saved.', flush=True)


if __name__ == '__main__':
    main()
