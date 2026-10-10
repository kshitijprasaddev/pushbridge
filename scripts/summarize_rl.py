"""Summarize the fixed-budget SAC ablation without selecting models."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pushbridge.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    out = args.run
    protocol = json.loads((out/'protocol.json').read_text())
    evaluation = json.loads((out/'evaluation/simulation_metrics.json').read_text())
    seeds = protocol['training_seeds']
    variants = ['scratch', 'demo_initialized']
    colors = {'scratch': '#d47726', 'demo_initialized': '#237f8d'}
    labels = {'scratch': 'SAC from scratch', 'demo_initialized': 'Personal-demo initialization + SAC'}
    curves, metadata = {}, {}
    summary = {'protocol': protocol, 'seed_aggregation': 'Each model faces the same 20 trials. Model seeds are replicates; these are not 60 independent task trials.', 'variants': {}}
    for variant in variants:
        names = [f'sac_{variant}_seed{s}' for s in seeds]
        curves[variant] = [json.loads((out/n/'learning_curve.json').read_text())['rows'] for n in names]
        metadata[variant] = [json.loads((out/n/'training_metrics.json').read_text()) for n in names]
        results = [evaluation['results'][n] for n in names]
        summary['variants'][variant] = {'seeds': seeds, 'successes_per_20_trials': [r['successes'] for r in results],
             'mean_success_rate_across_model_seeds': float(np.mean([r['success_rate'] for r in results])),
             'mean_final_distance_mm_across_model_seeds': float(np.mean([r['mean_final_distance_mm'] for r in results])),
             'p95_decision_ms_per_model': [r['p95_controller_ms'] for r in results],
             'sim_interaction_steps_per_model': [m['sim_interaction_steps'] for m in metadata[variant]],
             'total_training_control_steps_per_model': [m['total_training_control_steps'] for m in metadata[variant]],
             'training_seconds_including_validation_per_model': [m['training_seconds_including_validation'] for m in metadata[variant]]}
    summary['baselines'] = {name: {k: v for k, v in evaluation['results'][name].items() if k != 'runs'} for name in ['geometry', 'mpc', 'prior', 'replay', 'mpc_no_uncertainty']}
    bc_results = [evaluation['results'][f'bc_only_seed{s}'] for s in seeds]
    summary['imitation_only'] = {'seeds': seeds, 'successes_per_20_trials': [r['successes'] for r in bc_results],
              'mean_final_distance_mm_across_model_seeds': float(np.mean([r['mean_final_distance_mm'] for r in bc_results])),
              'p95_decision_ms_per_model': [r['p95_controller_ms'] for r in bc_results],
              'sim_training_interaction_steps': 0}
    write_json(out/'summary.json', summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for variant in variants:
        lengths = {len(c) for c in curves[variant]}
        if len(lengths) != 1:
            raise ValueError('Learning curves have different lengths; the experiment may still be running')
        x = np.array([r['sim_interaction_steps'] for r in curves[variant][0]])
        successes = np.array([[r['successes']/r['episodes']*100 for r in c] for c in curves[variant]])
        errors = np.array([[r['mean_final_distance_mm'] for r in c] for c in curves[variant]])
        for ax, data in zip(axes, [successes, errors]):
            ax.plot(x, data.mean(0), color=colors[variant], label=labels[variant])
            ax.fill_between(x, data.min(0), data.max(0), color=colors[variant], alpha=.18)
            for seed_data in data:
                ax.plot(x, seed_data, color=colors[variant], alpha=.18, linewidth=.8)
    axes[0].set(ylabel='Validation successes, %', ylim=(-5, 105))
    axes[1].set(ylabel='Mean validation final distance, mm')
    for ax in axes:
        ax.set(xlabel='Simulator interaction steps per model')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle(f"Learning curves: mean and range across {len(seeds)} training seeds; 10 validation trials", fontsize=11)
    for ext in ['png', 'svg']:
        fig.savefig(out/f'learning_curves.{ext}', dpi=180)
    plt.close(fig)

    rows = []
    for name in ['geometry', 'mpc', 'prior', 'replay', 'mpc_no_uncertainty']:
        r = summary['baselines'][name]
        rows.append(f'<tr><td>{name}</td><td>{r["successes"]}/20</td><td>{r["mean_final_distance_mm"]:.1f}</td><td>{r["p95_controller_ms"]:.2f}</td><td>0</td></tr>')
    for variant in variants:
        r = summary['variants'][variant]
        success = ', '.join(f'{n}/20' for n in r['successes_per_20_trials'])
        latency = ', '.join(f'{n:.2f}' for n in r['p95_decision_ms_per_model'])
        rows.append(f'<tr><td>{labels[variant]}</td><td>{success}</td><td>{r["mean_final_distance_mm_across_model_seeds"]:.1f}</td><td>{latency}</td><td>{protocol["interaction_steps_per_model"]:,} per model</td></tr>')
    bc = summary['imitation_only']
    bc_success = ', '.join(f'{n}/20' for n in bc['successes_per_20_trials'])
    bc_latency = ', '.join(f'{n:.2f}' for n in bc['p95_decision_ms_per_model'])
    rows.append(f'<tr><td>Personal-data imitation only</td><td>{bc_success}</td><td>{bc["mean_final_distance_mm_across_model_seeds"]:.1f}</td><td>{bc_latency}</td><td>0</td></tr>')
    initial_rates = {v: [c[0]['successes'] for c in curves[v]] for v in variants}
    initial_labels = metadata['demo_initialized'][0]['demo_initialization']['base_labels']
    times = [m['training_seconds_including_validation'] for v in variants for m in metadata[v]]
    forgetting = sum(min(r['successes'] for r in c[1:]) < c[0]['successes'] for c in curves['demo_initialized'])
    regressions = sum(c[-1]['successes'] < max(r['successes'] for r in c[:-1]) for c in curves['scratch'])
    data_finding = ('Data initialization improves mean final success at this budget in this run.'
                    if summary['variants']['demo_initialized']['mean_success_rate_across_model_seeds'] > summary['variants']['scratch']['mean_success_rate_across_model_seeds']
                    else 'Data initialization does not improve mean final success at this budget in this run.')
    failure_figures = []
    for video in sorted((out/'evaluation').glob('*_failure_*.mp4')):
        failure_figures.append(f'<p>{html.escape(video.stem)}</p><video controls muted preload="metadata" poster="evaluation/{html.escape(video.stem)}.jpg" src="evaluation/{html.escape(video.name)}"></video>')
    failure_html = '<h2>Inspect failures</h2><p>Recorded actions are replayed through the physical simulator, and the resulting states are checked against the saved evaluation trajectories.</p>'+''.join(failure_figures) if failure_figures else ''
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PushBridge - SAC, MPC and classical feedback</title>
<style>body{{font:17px/1.55 system-ui;max-width:1100px;margin:36px auto;padding:0 24px;background:#101a27;color:#eef2f6}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:38px}}a{{color:#95d8dc}}img,video{{max-width:100%;border-radius:8px}}.note{{background:#26364a;padding:18px;border-left:4px solid #75cbb0}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #3e5065}}.scroll{{overflow-x:auto}}</style>
<p style="border-bottom:1px solid #526653;padding:0 0 18px"><a href="../../index.html">Interactive submission report ↗</a></p><h1>Does personal data help reinforcement learning?</h1>
<p>SAC from scratch versus personal-demo actor initialization, compared with learned-model MPC and classical geometric feedback.</p>
<p class="note">State-based robot learning. Every controller sees object, tool and goal coordinates from MuJoCo. The real videos affect the dynamics model, demonstration prior and initialized SAC actor. Vision/language inputs are outside this experiment.</p>
<h2>Controlled comparison</h2><p>Two SAC variants, {len(seeds)} training seeds ({', '.join(map(str,seeds))}), {protocol['interaction_steps_per_model']:,} simulator interaction steps per model. Both use the same actor/critic architecture, reward, action bounds and training distribution. Validation uses seed 9012; the final fixed-budget checkpoints are evaluated once on twenty paired seed-2026 trials. Learning curves do not select checkpoints.</p>
<p>The imitation initializer uses {initial_labels} action labels from the three training recordings, with consistent rotation augmentation. It minimizes action error for the actor mean. Critics and replay learn entirely from robot simulation. Human video transitions are never treated as robot-physics experience. The held-out and validation recordings are excluded from imitation.</p>
<p>The initialized actor is also evaluated as an imitation-only baseline, before any simulator learning, for every seed. This isolates the value of RL beyond the video-trained actor.</p>
<img src="learning_curves.png" alt="SAC validation learning curves across training seeds">
<p>Before simulator learning, scratch models succeed in {html.escape(str(initial_rates['scratch']))} of ten validation trials per seed, and initialized actors in {html.escape(str(initial_rates['demo_initialized']))}. Shaded bands show the range across model seeds. A small three-seed study on one easy task cannot establish a general sample-efficiency advantage.</p>
<h2>Final paired outcomes</h2><div class="scroll"><table><tr><th>Controller</th><th>Successes per model</th><th>Mean final distance, mm</th><th>p95 decision, ms</th><th>Simulation training</th></tr>{''.join(rows)}</table></div>
<p>At the fixed budget, initialized SAC succeeds in {html.escape(str(summary['variants']['demo_initialized']['successes_per_20_trials']))} of twenty trials per seed; scratch SAC gets {html.escape(str(summary['variants']['scratch']['successes_per_20_trials']))}. Imitation alone gets {html.escape(str(bc['successes_per_20_trials']))}. Initialized SAC's mean final distance is {summary['variants']['demo_initialized']['mean_final_distance_mm_across_model_seeds']:.1f} mm, versus {summary['variants']['scratch']['mean_final_distance_mm_across_model_seeds']:.1f} mm from scratch and {bc['mean_final_distance_mm_across_model_seeds']:.1f} mm for imitation alone.</p>
<p>{data_finding} {forgetting}/{len(seeds)} initialized runs temporarily drop below their starting validation success, and {regressions}/{len(seeds)} scratch runs finish below an earlier validation peak. Learning curves matter alongside final results. MPC and classical feedback receive no simulator policy training; their final distances are {summary['baselines']['mpc']['mean_final_distance_mm']:.1f} and {summary['baselines']['geometry']['mean_final_distance_mm']:.1f} mm.</p>
<p>Each learned model faces the same twenty task trials. Repeating those trials across model seeds does not create sixty independent tasks. The goal tolerance is 18 mm after a one-second zero-action hold. The geometric baseline uses a hand-coded approach-and-push rule with proportional tip-position feedback and joint PD servos. MPC ranks candidate actions with its real-data dynamics ensemble and includes geometric candidates.</p>
<h2>What the comparison means</h2><p>SAC learns a fast feed-forward policy after additional simulator experience. MPC plans online from the video-fitted model without simulation training. Their final success rates therefore come with different data and compute budgets. Dense reward shaping uses progress toward the goal and toward a behind-object contact pose; this is explicit engineered task knowledge, shared by both SAC variants.</p>
<p>Reported budgets count agent decisions. Stable-success checks execute ten additional zero-action control ticks; exact total training-control counts and wall times are in the resource summary. Validation trajectories are extra evaluation, not training transitions. The {len(times)} training runs took approximately {min(times):.0f}-{max(times):.0f} seconds each on CPU, including validation.</p>
{failure_html}
<p>The original challenge permits world models and hand-coded approaches. RL is a useful extension for learning/sample-efficiency experiments; it does not make a state policy a VLA. A later vision-language-action experiment would need an image/language-conditioned policy, demonstration adaptation and a separate perception evaluation.</p>
<p><a href="evaluation/report.html">All controller videos, failures and per-trial outcomes</a> · <a href="summary.json">Summary and resource accounting</a> · <a href="protocol.json">Pre-training protocol</a> · <a href="../personal_five/report.html">Personal-video dynamics experiment</a></p>
<p>Implementation: <a href="https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/sac.html">Stable Baselines3 SAC</a>, following the <a href="https://arxiv.org/abs/1801.01290">SAC paper</a>. PPO and TD3 are possible follow-up baselines; neither was trained in this comparison.</p></html>'''
    (out/'report.html').write_text(page)
    print(f'Saved {out}/report.html')


if __name__ == '__main__':
    main()
