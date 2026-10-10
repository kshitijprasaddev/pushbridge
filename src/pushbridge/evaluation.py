import html
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .data import write_json
from .model import WorldModel
from .policy import Policy
from .sim import PushEnv
from .video import VideoWriter


def wilson(successes, total):
    p, z = successes/total, 1.96
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0, center-half), min(1, center+half)]


def make_trials(episodes, seed):
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(episodes):
        puck = rng.uniform([0.075, 0.065], [0.13, 0.145])
        goal = rng.uniform([0.21, 0.06], [0.25, 0.15])
        direction = (goal-puck)/np.linalg.norm(goal-puck)
        tool = puck-0.040*direction + rng.normal(0, 0.002, 2)
        trials.append({'puck': puck, 'goal': goal, 'tool': tool, 'friction': float(rng.choice([0.15, 0.25, 0.40]))})
    return trials


def evaluate(checkpoint, out, episodes=20, steps=120, seed=2026, render=True, modes=None, rl_models=None):
    if episodes < 1 or steps < 1:
        raise ValueError('episodes and steps must be positive')
    world = WorldModel.load(checkpoint)
    import torch
    torch.set_num_threads(2)
    rl_models = rl_models or {}
    modes = modes or ['replay', 'prior', 'geometry', 'mpc', 'mpc_no_uncertainty', *rl_models]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trials = make_trials(episodes, seed)
    results = {}
    for mode in modes:
        rl_policy = None
        if mode in rl_models:
            from .rl import SACPolicy
            rl_policy = SACPolicy(rl_models[mode])
        runs, decision_times = [], []
        for i, trial in enumerate(trials):
            env = PushEnv(friction=trial['friction'])
            state = env.reset(trial['puck'], trial['tool'], trial['goal'])
            policy = rl_policy or Policy(world, 'mpc' if mode == 'mpc_no_uncertainty' else mode, seed+i,
                                         uncertainty_weight=0 if mode == 'mpc_no_uncertainty' else 0.5)
            writer = None
            if render and i == 0:
                writer = VideoWriter(out/f'{mode}.mp4', 10, (960, 640))
                if not writer.isOpened():
                    raise RuntimeError('Video encoder unavailable')
            history, action_history, succeeded = [state.copy()], [], False
            try:
                for step in range(steps):
                    tick = time.perf_counter()
                    action = policy.act(state, trial['goal'])
                    decision_times.append((time.perf_counter()-tick)*1000)
                    state = env.step(action)
                    history.append(state.copy())
                    action_history.append(action.copy())
                    distance = np.linalg.norm(state[2:]-trial['goal'])
                    if writer:
                        frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                        banner = f'{mode} | step {step+1} | goal distance {distance*1000:.1f} mm'
                        cv2.rectangle(frame, (0, 0), (960, 82), (24, 29, 39), -1)
                        cv2.putText(frame, banner, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        if rl_policy:
                            label = 'STATE POLICY | MuJoCo ground-truth observations'
                        else:
                            label = 'SYNTHETIC SMOKE TEST - NOT APPLICANT DATA' if world.checkpoint['source'] != 'personal' else 'PERSONAL-DATA EXPERIMENT | MuJoCo ground-truth observations'
                        cv2.putText(frame, label, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 200, 245), 1)
                        writer.write(frame)
                        if step == 0:
                            cv2.imwrite(str(out/f'{mode}_poster.jpg'), frame)
                            if mode == 'mpc' or not (out/'simulation.png').exists():
                                cv2.imwrite(str(out/'simulation.png'), frame)
                    if distance < 0.018:
                        for _ in range(10):
                            state = env.step([0, 0])
                            history.append(state.copy())
                            action_history.append(np.zeros(2))
                            if writer:
                                settled = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                                cv2.rectangle(settled, (0, 0), (960, 82), (24, 29, 39), -1)
                                cv2.putText(settled, f'{mode} | holding tool still | distance {1000*np.linalg.norm(state[2:]-trial["goal"]):.1f} mm', (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(settled, label, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 200, 245), 1)
                                writer.write(settled)
                        succeeded = bool(np.linalg.norm(state[2:]-trial['goal']) < 0.018)
                        if succeeded:
                            break
                    if np.any(state[2:] < 0) or np.any(state[2:] > [0.297, 0.210]):
                        break
            finally:
                if writer:
                    writer.release()
                env.close()
            runs.append({'trial': i, 'success': succeeded, 'steps': step+1,
                         'final_distance_mm': float(np.linalg.norm(state[2:]-trial['goal'])*1000),
                         'friction': trial['friction'], 'initial_puck': trial['puck'].tolist(),
                         'initial_tool': trial['tool'].tolist(), 'goal': trial['goal'].tolist()})
            np.savez_compressed(out/f'{mode}_{i:03d}.npz', state=np.array(history), action=np.array(action_history), time=np.arange(len(history))*0.1, goal=trial['goal'])
        n_success = sum(r['success'] for r in runs)
        results[mode] = {'successes': n_success, 'episodes': episodes, 'success_rate': n_success/episodes,
                         'success_rate_95pct_wilson_interval': wilson(n_success, episodes),
                         'mean_final_distance_mm': float(np.mean([r['final_distance_mm'] for r in runs])),
                         'p50_controller_ms': float(np.median(decision_times)), 'p95_controller_ms': float(np.quantile(decision_times, 0.95)),
                         'runs': runs}
        print(f'{mode}: {n_success}/{episodes} successes, mean final error {results[mode]["mean_final_distance_mm"]:.1f} mm', flush=True)
    report = {'source': world.checkpoint['source'], 'seed': seed, 'success_threshold_mm': 18,
              'max_steps': steps, 'state_observation': 'MuJoCo ground truth', 'physics': 'contact and friction; no object attachment or scripted object movement',
              'protocol': 'Paired starts, goals and friction across controllers. Success persists after 1 second with zero requested tool displacement.',
              'results': results}
    prediction_path = Path(checkpoint).parent/'prediction_metrics.json'
    if prediction_path.exists():
        prediction_report = json.loads(prediction_path.read_text())
        report['held_out_prediction_metrics'] = prediction_report['test']
        report['coordinate_units'] = prediction_report.get('coordinate_units', 'calibrated tabletop metres')
        write_json(out/'prediction_metrics.json', prediction_report)
    write_json(out/'simulation_metrics.json', report)
    make_report(report, out, render)
    return report


def make_report(report, out, render):
    rows, videos = [], []
    for mode, result in report['results'].items():
        lo, hi = result['success_rate_95pct_wilson_interval']
        rows.append(f'<tr><td>{html.escape(mode)}</td><td>{result["successes"]}/{result["episodes"]}</td><td>{lo:.0%}-{hi:.0%}</td><td>{result["mean_final_distance_mm"]:.1f}</td><td>{result["p95_controller_ms"]:.1f}</td></tr>')
        if render:
            videos.append(f'<figure><figcaption>{html.escape(mode)} · first paired trial</figcaption><video controls muted preload="metadata" poster="{mode}_poster.jpg" src="{mode}.mp4"></video></figure>')
    label = 'SYNTHETIC SOFTWARE CHECK - does not satisfy the applicant-data requirement' if report['source'] != 'personal' else 'PERSONAL RECORDINGS - declared by the operator'
    prediction = report.get('held_out_prediction_metrics')
    prediction_html = ''
    if prediction:
        prediction_html = f'<h2>Held-out state predictions</h2><p>One-step mean error: <b>{prediction["mean_error_mm"]:.2f} mm</b> across {prediction["transitions"]} transitions. Zero-motion baseline: {prediction["zero_motion_error_mm"]:.2f} mm; fixed contact baseline: {prediction["analytic_contact_error_mm"]:.2f} mm. Moving-transition error: {prediction["moving_mean_error_mm"]:.2f} mm.</p><p><a href="prediction_metrics.json">Prediction metrics, split IDs and modelling limitations</a></p>' if prediction['moving_mean_error_mm'] is not None else '<p><a href="prediction_metrics.json">Prediction metrics and split IDs</a></p>'
    coordinate_note = ''
    if report.get('coordinate_units'):
        coordinate_note = f'<p>Prediction distances use {html.escape(report["coordinate_units"])}. Simulation distances use the simulator\'s metre scale.</p>'
    content = f'''<!doctype html><html><meta charset="utf-8"><title>PushBridge results</title>
<style>body{{font:17px system-ui;background:#101a27;color:#eee;max-width:1100px;margin:40px auto;padding:0 24px}}.source{{background:#29354b;padding:18px;border-left:5px solid #e7b358}}table{{border-collapse:collapse;width:100%;margin:30px 0}}td,th{{padding:14px;text-align:left;border-bottom:1px solid #42506a}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0}}video{{width:100%;margin-top:12px}}a{{color:#9fdbda}}@media(max-width:750px){{.videos{{grid-template-columns:1fr}}}}</style>
<h1>PushBridge</h1><p>Phone-video demonstrations → contact dynamics → simulated arm.</p><p class="source">{label}</p>
<p>{report['protocol']} The goal tolerance is 18 mm. The robot observes simulator states.</p>
<table><tr><th>Controller</th><th>Successes</th><th>95% interval</th><th>Mean final error, mm</th><th>p95 decision, ms</th></tr>{''.join(rows)}</table>
{prediction_html}
{coordinate_note}
<p>Geometry is a hand-coded baseline. MPC includes geometric candidates; any improvement must be judged against that baseline. Comparing MPC with and without its uncertainty penalty isolates that penalty. This result alone does not establish real-to-sim transfer or generalisation.</p>
<p><a href="simulation_metrics.json">Full metrics and per-trial records</a></p><div class="videos">{''.join(videos)}</div></html>'''
    (Path(out)/'report.html').write_text(content)
