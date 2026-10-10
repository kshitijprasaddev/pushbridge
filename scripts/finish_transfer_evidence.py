"""Preserve all transfer outcomes; add the explicit compact-control baseline."""
from pathlib import Path
import json
import cv2
import numpy as np

from pushbridge.data import write_json
from pushbridge.embodiment import CompactGeometry, ContactAdapter
from pushbridge.evaluation import make_trials, wilson
from pushbridge.rl import SACPolicy
from pushbridge.transfer import run_trial
from pushbridge.video import VideoWriter


def main():
    out = Path('results/embodiment_transfer')
    report = json.loads((out/'evaluation.json').read_text())
    protocol = report['protocol']
    protocol['followup_baseline_note'] = ('CompactGeometry was implemented after the initial comparison, inspected on validation seed 7131, '
                                         'and retained as an additional baseline. Original results are preserved.')
    baseline = 'compact-aware classical feedback (follow-up baseline)'
    if baseline not in protocol['baselines']:
        protocol['baselines'].append(baseline)
    write_json(out/'protocol.json', protocol)
    rows, times = [], []
    for i, trial in enumerate(make_trials(20, 2027)):
        writer = VideoWriter(out/'geometry_compact.mp4', 10, (960, 640)) if i == 0 else None
        try:
            row, trace, poster = run_trial(CompactGeometry(), trial, writer=writer, label='Compact 3R | compact-aware classical control')
        finally:
            if writer:
                writer.release()
        row['trial'] = i
        rows.append(row)
        times.extend(trace['decision_ms'])
        np.savez_compressed(out/f'geometry_compact_{i:03d}.npz', **trace)
        if poster is not None:
            cv2.imwrite(str(out/'geometry_compact.jpg'), poster)
    n = sum(r['success'] for r in rows)
    report['results']['geometry_compact'] = {'successes': n, 'episodes': 20, 'wilson_95pct': wilson(n, 20),
                                           'mean_final_distance_mm': float(np.mean([r['final_distance_mm'] for r in rows])),
                                           'p95_decision_ms': float(np.quantile(times, .95)), 'runs': rows,
                                           'development': protocol['followup_baseline_note']}
    failures = []
    for mode in ['source_direct_seed1', 'posttrained_seed2']:
        trial_row = next(r for r in report['results'][mode]['runs'] if not r['success'])
        trial = {k: np.asarray(trial_row[k]) for k in ['goal', 'initial_puck', 'initial_tool']}
        trial = {'goal': trial['goal'], 'puck': trial['initial_puck'], 'tool': trial['initial_tool'], 'friction': trial_row['friction']}
        seed = int(mode[-1])
        source = out/f'posttrained_seed{seed}/sac.zip' if mode.startswith('posttrained') else Path(f'results/rl_comparison/sac_demo_initialized_seed{seed}/sac.zip')
        policy = SACPolicy(source)
        if mode.startswith('posttrained'):
            policy = ContactAdapter(policy)
        name = f'{mode}_failure_{trial_row["trial"]:03d}'
        writer = VideoWriter(out/f'{name}.mp4', 10, (960, 640))
        try:
            row, trace, poster = run_trial(policy, trial, writer=writer, label=f'Failure analysis | {mode}')
        finally:
            writer.release()
        original = np.load(out/f'{mode}_{trial_row["trial"]:03d}.npz')
        difference = float(np.abs(trace['state']-original['state']).max())
        original.close()
        if difference > 1e-7:
            raise RuntimeError('Rendered failure did not reproduce recorded physics')
        cv2.imwrite(str(out/f'{name}.jpg'), poster)
        failures.append({'mode': mode, 'trial': trial_row['trial'], 'video': name+'.mp4', 'poster': name+'.jpg',
                         'max_recorded_state_difference_m': difference, 'final_distance_mm': row['final_distance_mm']})
    report['verified_failures'] = failures
    write_json(out/'evaluation.json', report)
    print(f'Compact-aware classical baseline: {n}/20; failures verified: {failures}', flush=True)


if __name__ == '__main__':
    main()
