"""Build plots and a local review page from an already evaluated experiment.

Run from the repository root with the optional `analysis` dependencies installed.
This script reads saved metrics and weights; it does not fit or select models.
"""
import argparse
import html
import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pushbridge.data import load_episodes, transitions, write_json
from pushbridge.model import WorldModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    out = args.run
    prediction = json.loads((out/'model/prediction_metrics.json').read_text())
    simulation = json.loads((out/'evaluation/simulation_metrics.json').read_text())
    manifest, episodes = load_episodes(args.dataset)
    world = WorldModel.load(out/'model/world_model.pt')
    split = prediction['split']
    role = {name: group.replace('_ids', '') for group, names in split.items() for name in names}
    test = prediction['test']
    mpc_result = simulation['results']['mpc']
    geometry_result = simulation['results']['geometry']
    trial_count = mpc_result['episodes']
    clip_count = len(episodes)
    colors = {'train': '#176f83', 'validation': '#d77d19', 'test': '#9851a5'}

    # Show observed trajectories and one-step predictions on the held-out clip.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ep in episodes:
        xy = ep['state'][:, 2:]*1000
        axes[0].plot(*xy.T, label=f"{ep['id']} ({role[ep['id']]})", color=colors[role[ep['id']]], alpha=.8)
        axes[0].scatter(*xy[0], color=colors[role[ep['id']]], s=18)
    axes[0].add_patch(plt.Circle((225, 105), 20, fill=False, color='#666', linestyle='--'))
    axes[0].set(xlabel='Normalised x, mm', ylabel='Normalised y, mm', title=f'{clip_count} recorded object paths')
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=8, loc='lower left')
    held_out = next(ep for ep in episodes if ep['id'] in split['test_ids'])
    state, action, delta, indices = transitions(held_out)
    estimate, _ = world.predict(state[:, :2]-state[:, 2:], action)
    axes[1].plot(held_out['time'][indices], np.linalg.norm(delta, axis=1)*1000, label='Tracked object motion', color='#176f83')
    axes[1].plot(held_out['time'][indices], np.linalg.norm(estimate, axis=1)*1000, label='One-step model prediction', color='#d77d19')
    axes[1].set(xlabel='Time within selected segment, s', ylabel='Displacement per 0.1 s, mm', title=f"Held-out clip: {held_out['id']}")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Circle-relative coordinates: target radius maps to 20 mm in simulation', fontsize=11)
    for suffix in ['png', 'svg']:
        fig.savefig(out/f'predictions.{suffix}', dpi=180)
    plt.close(fig)

    modes = list(simulation['results'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    rates = [simulation['results'][m]['success_rate']*100 for m in modes]
    bounds = np.array([simulation['results'][m]['success_rate_95pct_wilson_interval'] for m in modes])*100
    axes[0].bar(range(len(modes)), rates, color='#176f83', alpha=.8)
    axes[0].errorbar(range(len(modes)), rates, yerr=[np.array(rates)-bounds[:, 0], bounds[:, 1]-rates], fmt='none', color='#333', capsize=4)
    axes[0].set(ylim=(-5, 110), ylabel='Success rate, %', title=f'{trial_count} paired trials; Wilson 95% intervals')
    for j, mode in enumerate(modes):
        errors = np.array([r['final_distance_mm'] for r in simulation['results'][mode]['runs']])
        axes[1].scatter(j+np.linspace(-.12, .12, len(errors)), errors, s=18, alpha=.65, color='#176f83')
        axes[1].scatter(j, errors.mean(), s=70, marker='_', color='#b05125', linewidths=3)
    axes[1].axhline(18, color='#666', linestyle='--', label='Success tolerance')
    axes[1].set(ylabel='Final simulator goal distance, mm', title='All final outcomes; orange mark is the mean')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(modes)), [m.replace('mpc_no_uncertainty', 'mpc without\nuncertainty') for m in modes], rotation=20, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=.2)
    for suffix in ['png', 'svg']:
        fig.savefig(out/f'controllers.{suffix}', dpi=180)
    plt.close(fig)

    # A descriptive paired bootstrap keeps every controller's trial pairing.
    geometry = np.array([r['final_distance_mm'] for r in simulation['results']['geometry']['runs']])
    mpc = np.array([r['final_distance_mm'] for r in simulation['results']['mpc']['runs']])
    difference = geometry-mpc
    rng = np.random.default_rng(2026)
    interval = np.quantile(rng.choice(difference, (10000, len(difference)), replace=True).mean(1), [.025, .975])
    write_json(out/'paired_comparison.json', {'quantity': 'geometry minus MPC final distance, mm; positive favours MPC',
               'mean': float(difference.mean()), 'bootstrap_95pct_interval': interval.tolist(),
               'paired_trials': len(difference), 'bootstrap_samples': 10000, 'seed': 2026,
               'limitation': 'Descriptive interval for this small seeded task suite; not broad generalisation evidence.'})

    # Export three inspectable tracking frames per clip from the saved tracks.
    sheets, data_rows = [], []
    def url(path):
        return html.escape(os.path.relpath(path, out).replace(os.sep, '/'), quote=True)
    for i, ep in enumerate(episodes):
        folder = args.dataset/f'{i:03d}_diagnostics'
        with np.load(folder/'extracted_task.npz', allow_pickle=False) as tracks:
            selected = np.linspace(0, len(tracks['video_time_s'])-1, 3).astype(int)
            cap = cv2.VideoCapture(str(args.raw/ep['raw_name']))
            frames = []
            try:
                for j in selected:
                    t = float(tracks['video_time_s'][j])
                    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot read {ep['raw_name']} at {t:.1f}s")
                    scale = 360/frame.shape[1]
                    frame = cv2.resize(frame, (360, 480))
                    for key, color in [('object_px', (50, 240, 70)), ('finger_px', (255, 190, 10)), ('target_px', (40, 220, 255))]:
                        xy = tuple(np.round(tracks[key][j]*scale).astype(int))
                        cv2.drawMarker(frame, xy, color, cv2.MARKER_CROSS, 18, 2)
                    cv2.rectangle(frame, (0, 0), (360, 43), (24, 29, 39), -1)
                    cv2.putText(frame, f"{ep['id']} | {t:.1f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                    frames.append(frame)
            finally:
                cap.release()
            sheets.append(np.hstack(frames))
        start, end = ep['tracking']['continuous_segment_s']
        data_rows.append(f'<tr><td>{html.escape(ep["raw_name"])}</td><td>{role[ep["id"]]}</td><td>{start:.1f}-{end:.1f}s</td><td>{ep["transitions"]}</td><td>{ep["moving_transitions"]}</td><td><a href="{url(folder/"tracking_preview.mp4")}">Tracking</a></td></tr>')
    cv2.imwrite(str(out/'dataset_preview.jpg'), np.vstack(sheets))

    control_rows = []
    for mode, r in simulation['results'].items():
        control_rows.append(f'<tr><td>{html.escape(mode)}</td><td>{r["successes"]}/{r["episodes"]}</td><td>{r["mean_final_distance_mm"]:.1f}</td><td>{r["p95_controller_ms"]:.2f}</td></tr>')
    total = sum(ep['transitions'] for ep in episodes)
    moving = sum(ep['moving_transitions'] for ep in episodes)
    control_comparison = f'MPC succeeds in {mpc_result["successes"]}/{trial_count} trials; geometry succeeds in {geometry_result["successes"]}/{trial_count}.'
    interval_note = 'The interval includes zero, so this does not establish a control advantage.' if interval[0] <= 0 <= interval[1] else 'The interval describes this small paired trial suite; broader task coverage is needed.'
    lo, hi = mpc_result['success_rate_95pct_wilson_interval']
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PushBridge - personal data experiment</title>
<style>body{{font:17px/1.55 system-ui;margin:36px auto;padding:0 24px;max-width:1080px;background:#101a27;color:#eef2f6}}h1{{font-size:40px;line-height:1.15}}h2{{margin-top:42px}}a{{color:#95d8dc}}.note{{background:#26364a;padding:18px;border-left:4px solid #75cbb0}}.numbers{{display:flex;gap:32px;flex-wrap:wrap;margin:24px 0}}.numbers b{{display:block;font-size:30px}}video,img{{max-width:100%;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #3e5065}}.scroll{{overflow-x:auto}}details{{margin:20px 0}}summary{{cursor:pointer;color:#95d8dc}}</style>
<p style="border-bottom:1px solid #526653;padding:0 0 18px"><a href="../../index.html">Interactive submission report ↗</a></p><h1>PushBridge</h1><p>{clip_count} home recordings → learned contact dynamics → a physical MuJoCo arm.</p>
<p><a href="../rl_comparison/report.html">New: SAC, imitation, MPC and classical feedback comparison</a></p>
<p class="note">Personal videos supplied by the user. The model learns from tracked finger and cap positions; the robot moves the object through simulated contact. This is a state dynamics world model.</p>
<div class="numbers"><div><b>{clip_count} clips</b>{len(split['train_ids'])} training · {len(split['validation_ids'])} validation · {len(split['test_ids'])} test</div><div><b>{test['mean_error_mm']:.2f} mm</b>held-out one-step error</div><div><b>{mpc_result['successes']}/{trial_count}</b>learned planner successes</div></div>
<video controls muted preload="metadata" poster="evaluation/mpc_poster.jpg" src="evaluation/mpc.mp4"></video><p>The learned planner's first paired trial. The tolerance is 18 mm after a one-second hold. The arm observes MuJoCo ground-truth states.</p>
<h2>The data</h2><p>{total} usable transitions at 10 Hz, including {moving} with object motion above 1 mm in the normalised frame. The extractor retains each video's longest uninterrupted segment with both markers and an acceptable target-circle fit. Later occluded segments are excluded. Source hashes and original selected timestamps are preserved.</p>
<div class="scroll"><table><tr><th>Clip</th><th>Split</th><th>Retained segment</th><th>Transitions</th><th>Moving</th><th>Preview</th></tr>{''.join(data_rows)}</table></div>
<details><summary>Inspect tracked frames from all five clips</summary><p>Green: cap. Blue: finger marker. Yellow: fitted target centre.</p><img loading="lazy" src="dataset_preview.jpg" alt="Three tracked frames from each of five personal videos"></details>
<p>The fitted circle anchors image position and approximate scale. Its radius maps to a chosen 20 mm simulator radius. Prediction distances below are normalised simulator distances; physical dimensions in the videos were not measured. This compensates approximate translation and zoom, rather than full camera perspective or rotation.</p>
<h2>What the model learned</h2><p>Three small contact-gated models learn displacement gains and effective contact range. Rotation equivariance is built into their relative-state inputs and vector outputs. Training bootstraps whole training clips; validation selects checkpoints. The test clip is excluded from fitting, the demonstration prior and checkpoint selection.</p>
<img src="predictions.png" alt="Normalised object paths and held-out displacement predictions">
<p>Held-out mean vector error: <b>{test['mean_error_mm']:.2f} mm</b>; zero-motion baseline: {test['zero_motion_error_mm']:.2f} mm; fixed analytic-contact baseline: {test['analytic_contact_error_mm']:.2f} mm. Moving-transition error: {test['moving_mean_error_mm']:.2f} mm. Five-step and ten-step open-loop errors grow to {test['5_step_open_loop_error_mm']:.2f} and {test['10_step_open_loop_error_mm']:.2f} mm. There is only one test recording.</p>
<h2>Control comparison</h2><p>All five controllers receive identical starts, goals and three possible floor-friction values across twenty seeded trials. Geometry is a hand-coded baseline. MPC ranks demonstration and geometric proposals using learned dynamics over four steps, with an empirical uncertainty penalty.</p>
<div class="scroll"><table><tr><th>Controller</th><th>Successes</th><th>Mean final distance, mm</th><th>p95 decision, ms</th></tr>{''.join(control_rows)}</table></div>
<img src="controllers.png" alt="Paired controller success rates and every trial's final distance">
<p>{control_comparison} The mean final-distance difference (geometry minus MPC) is {difference.mean():.2f} mm; the descriptive paired bootstrap interval is {interval[0]:.2f} to {interval[1]:.2f} mm. {interval_note} MPC's Wilson success-rate interval is approximately {lo:.0%}-{hi:.0%} for this sampled suite.</p>
<h2>What worked and what failed</h2><p>Circle-relative coordinates made the cropped-paper recordings usable. Compression shifted the cap toward magenta and made thin black ink difficult to detect; revised hue limits, low-saturation circle detection, outline coverage checks and regression tests fixed observed identity failures. Occlusions still shorten the usable data. Raw action replay failed every simulation trial; adapting contact and direction matters.</p>
<p>One-step predictions improve on both prediction baselines on this test clip, but longer rollouts drift. The simple task is already solved by the geometric controller and the demonstration prior. Geometric MPC proposals carry substantial task knowledge. Five clips on one surface and object, approximate camera correction, different finger/tip contacts and simulator-state observations limit generalisation claims. No real robot, VLA, image encoder or generated-video model is used.</p>
<p><a href="evaluation/report.html">Every controller video and per-trial outcomes</a> · <a href="model/prediction_metrics.json">Prediction metrics and split IDs</a> · <a href="paired_comparison.json">Paired comparison</a> · <a href="{url(args.dataset/'manifest.json')}">Dataset manifest and hashes</a> · <a href="../second_personal_clip/report.html">Earlier hand-coded video retargeting demo</a></p></html>'''
    (out/'report.html').write_text(page)
    print(f'Saved {out}/report.html and figures')


if __name__ == '__main__':
    main()
