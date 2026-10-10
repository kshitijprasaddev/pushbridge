"""Verify the included evidence without training or rendering.

This checks provenance and internal consistency. It does not independently
establish human authorship, tracker accuracy, or hardware performance.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {'href', 'src', 'poster'} and value:
                url = urlsplit(value)
                if not url.scheme and not url.netloc and url.path:
                    self.paths.add(unquote(url.path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Optional JSON audit record')
    args = parser.parse_args()
    checks, hashes = [], {}
    manifest = read('data/processed/personal_five/manifest.json')
    for episode in manifest['episodes']:
        path = ROOT/'data/raw/personal_circle'/episode['raw_name']
        actual = digest(path)
        require(actual == episode['raw_sha256'], f'Publication recording hash changed: {path.name}')
        hashes[str(path.relative_to(ROOT))] = actual
    checks.append('All five published recording hashes match the extraction manifest.')

    publication = read('data/raw/personal_circle/publication_verification.json')
    from prepare_public_recordings import metadata_keys
    for item in publication['files']:
        require(item['decoded_video_sha256'] == item['original_decoded_video_sha256'], f'Published video pixels changed: {item["file"]}')
        require(digest(ROOT/'data/raw/personal_circle'/item['file']) == item['publication_sha256'], f'Publication verification hash differs: {item["file"]}')
        keys = metadata_keys(ROOT/'data/raw/personal_circle'/item['file'])
        require(not any('location' in k.lower() or 'gps' in k.lower() for k in keys), f'Location metadata remains: {item["file"]}')
    checks.append('Published videos have no GPS metadata; publication records verify identical decoded video frames.')

    split = read('results/personal_five/model/prediction_metrics.json')['split']
    sets = [set(values) for values in split.values()]
    require(len(sets) == 3 and sum(map(len, sets)) == 5, 'Expected a five-clip, three-way split')
    require(all(not a & b for i, a in enumerate(sets) for b in sets[i+1:]), 'Whole-clip split overlaps')
    checks.append('Training, validation and test recordings are disjoint.')

    evaluation = read('results/precision_finish/evaluation.json')
    protocol = read('results/precision_finish/protocol.json')
    require(evaluation['protocol'] == protocol, 'Evaluation protocol differs from the frozen protocol')
    xml = ET.parse(ROOT/'src/pushbridge/assets/push.xml').getroot()
    goal_radius = float(xml.find(".//site[@name='goal']").get('size').split()[0])*1000
    puck_radius = float(xml.find(".//geom[@name='puck_geom']").get('size').split()[0])*1000
    require(goal_radius == protocol['goal_radius_mm'] and puck_radius == protocol['puck_radius_mm'], 'Physical radii changed')
    require(goal_radius-puck_radius == protocol['fit_distance_mm'] == 8, 'Containment boundary is incorrect')
    for seed, recorded in protocol['checkpoints'].items():
        paths = {
            f'results/rl_comparison/sac_demo_initialized_seed{seed}/sac.zip': recorded['source_sha256'],
            f'results/embodiment_transfer/posttrained_seed{seed}/sac.zip': recorded['posttrained_sha256'],
        }
        for relative, expected in paths.items():
            actual = digest(ROOT/relative)
            require(actual == expected, f'Frozen checkpoint changed: {relative}')
            hashes[relative] = actual
    checks.append('Physical target dimensions and all six frozen checkpoint hashes match the protocol.')

    traces, holds = 0, 0
    for suite, values in evaluation['suites'].items():
        for name, result in values['results'].items():
            require(sum(row['success'] for row in result['runs']) == result['successes'], f'Success count differs: {name}')
            distances = []
            for row in result['runs']:
                path = ROOT/f'results/precision_finish/{suite}/{name}_{row["trial"]:03d}.npz'
                with np.load(path) as trace:
                    state, action, phase, joint = (trace[key] for key in ['state', 'action', 'phase', 'joint'])
                    require(len(state) == len(action)+1 == len(phase) == len(joint), f'Trace length differs: {path}')
                    require(np.isfinite(state).all() and np.isfinite(joint).all(), f'Non-finite trace: {path}')
                    error = np.linalg.norm(state[-1, 2:]-trace['goal'])*1000
                    require(abs(error-row['final_distance_mm']) < 1e-8, f'Endpoint differs: {path}')
                    require(abs((8-error)-row['final_clearance_mm']) < 1e-8, f'Clearance differs: {path}')
                    require(row['decisions'] <= 120, f'Decision budget exceeded: {path}')
                    require(row['adapter']['max_command_joint_speed_rad_s'] <= .25+1e-12, f'Command speed exceeded: {path}')
                    if row['success']:
                        hold = state[-10:]
                        require(np.all(phase[-10:] == 'verify'), f'Success without a complete hold: {path}')
                        require(np.all(np.linalg.norm(hold[:, 2:]-trace['goal'], axis=1)*1000 <= 8), f'Object outside during hold: {path}')
                        require(np.all(action[-10:] == 0), f'Non-stationary hold command: {path}')
                        holds += 1
                    distances.append(error)
                    traces += 1
            require(abs(np.mean(distances)-result['mean_final_distance_mm']) < 1e-8, f'Mean differs: {name}')
    checks.append(f'All {traces} precision traces match their endpoints, counts and command limits; {holds} successful holds fit throughout.')

    dashboard = read('dashboard/data.json')
    require(dashboard['precision'] == evaluation, 'Dashboard precision metrics differ from evidence')
    source = (ROOT/'dashboard/data.js').read_text()
    require(json.loads(source.removeprefix('window.PUSHBRIDGE_DATA = ').rstrip(';\n')) == dashboard, 'Offline data differs from JSON')
    for name, cases in dashboard['precision_traces'].items():
        for trial, displayed in enumerate(cases):
            with np.load(ROOT/f'results/precision_finish/paired/{name}_{trial:03d}.npz') as trace:
                require(all(displayed[key] == trace[key].tolist() for key in displayed), f'Displayed trace differs: {name}, task {trial}')
    checks.append('Dashboard metrics and all 360 displayed trajectories exactly match saved evidence.')

    links = Links()
    links.feed((ROOT/'index.html').read_text())
    for relative in links.paths:
        require((ROOT/relative).is_file(), f'Missing report asset: {relative}')
    checks.append(f'All {len(links.paths)} static report file links exist.')
    text_extensions = {'.md', '.html', '.js', '.css', '.py', '.json', '.svg', '.yml', '.toml'}
    ignored = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', '.venv', 'node_modules'}
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix in text_extensions and not ignored.intersection(path.relative_to(ROOT).parts):
            require(chr(0x2014) not in path.read_text(), f'Em dash remains: {path.relative_to(ROOT)}')
    checks.append('No em dashes remain in source, documentation or text artifacts.')
    audit = {'status': 'passed', 'checks': checks, 'precision_traces_checked': traces,
             'successful_holds_checked': holds, 'sha256': hashes,
             'scope': 'Included evidence consistency, not an independent rerun of every experiment or a hardware test'}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    for check in checks:
        print(f'PASS: {check}')


if __name__ == '__main__':
    main()
