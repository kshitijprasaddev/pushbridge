"""Export + alternating paired latency measurements + physical trace checks."""
import argparse
import platform
from pathlib import Path
import time

import numpy as np
import torch

from pushbridge.data import sha256, write_json
from pushbridge.evaluation import make_trials
from pushbridge.fast_policy import DirectTorchActor, FastActor, export_actor
from pushbridge.rl import SACPolicy
from pushbridge.sim import PushEnv
from pushbridge.transfer import run_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('results/rl_comparison/sac_demo_initialized_seed0/sac.zip'))
    parser.add_argument('--out', type=Path, default=Path('results/policy_speed'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    protocol = {'model_sha256': sha256(args.model), 'thread_count_torch': 2,
                'warmup_calls_per_method': 100, 'blocks': 30, 'calls_per_method_per_block': 100,
                'order': 'alternate methods by block, fixed observations shuffled by seed 871',
                'timed_scope': 'Full policy.act(state, goal), including observation encoding and action norm clipping; no physics/rendering',
                'trace_check_seed': 2026, 'trace_check_episodes': 20,
                'method': 'Float32 deterministic actor export, no retraining/quantisation',
                'platform': platform.platform(), 'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__}
    write_json(args.out/'protocol.json', protocol)
    standard = SACPolicy(args.model)
    exported = export_actor(standard.agent, args.out/'actor.npz')
    fast = FastActor(args.out/'actor.npz')
    # Existing evaluation observations are used for equivalence/timing only.
    # Nothing is fitted to evaluation labels or test outcomes.
    cases = []
    for p in sorted(Path('results/rl_comparison/evaluation').glob('sac_demo_initialized_seed0_*.npz')):
        with np.load(p) as d:
            cases.extend((s.copy(), d['goal'].copy()) for s in d['state'])
    rng = np.random.default_rng(871)
    rng.shuffle(cases)
    reference = np.array([standard.act(s, g) for s, g in cases])
    optimized = np.array([fast.act(s, g) for s, g in cases])
    agreement = np.linalg.norm(reference-optimized, axis=1)
    direct = DirectTorchActor(args.out/'actor.npz')
    methods = {'sb3_predict': standard, 'torch_direct': direct, 'numpy_actor': fast}
    for policy in methods.values():
        for s, g in cases[:100]:
            policy.act(s, g)
    timings, block_medians = {n: [] for n in methods}, {n: [] for n in methods}
    for block in range(30):
        names = list(methods) if block % 2 == 0 else list(reversed(methods))
        selected = [cases[(block*100+i) % len(cases)] for i in range(100)]
        for name in names:
            elapsed = []
            for s, g in selected:
                tick = time.perf_counter_ns()
                methods[name].act(s, g)
                elapsed.append((time.perf_counter_ns()-tick)/1e6)
            timings[name].extend(elapsed)
            block_medians[name].append(float(np.median(elapsed)))
    latency = {name: {'calls': len(t), 'p50_ms': float(np.median(t)), 'p95_ms': float(np.quantile(t, .95)),
                      'block_medians_ms': block_medians[name]} for name, t in timings.items()}
    ratios = np.array(block_medians['sb3_predict'])/np.array(block_medians['numpy_actor'])
    bootstrap = rng.choice(ratios, (10000, len(ratios)), replace=True).mean(axis=1)
    traces, exact_traces = [], []
    for i, trial in enumerate(make_trials(20, 2026)):
        row_a, trace_a, _ = run_trial(standard, trial, env_class=PushEnv)
        row_b, trace_b, _ = run_trial(fast, trial, env_class=PushEnv)
        row_c, trace_c, _ = run_trial(direct, trial, env_class=PushEnv)
        length = min(len(trace_a['state']), len(trace_b['state']))
        error = np.linalg.norm(trace_a['state'][:length]-trace_b['state'][:length], axis=1)
        row = {'trial': i, 'reference_success': row_a['success'], 'optimized_success': row_b['success'],
               'reference_control_steps': len(trace_a['action']), 'optimized_control_steps': len(trace_b['action']),
               'max_tool_and_object_position_vector_difference_mm': float(error.max()*1000),
               'reference_final_mm': row_a['final_distance_mm'], 'optimized_final_mm': row_b['final_distance_mm']}
        traces.append(row)
        exact_traces.append({'trial': i, 'success': row_c['success'],
                             'bit_identical_actions': bool(np.array_equal(trace_a['action'], trace_c['action'])),
                             'bit_identical_states': bool(np.array_equal(trace_a['state'], trace_c['state']))})
        np.savez_compressed(args.out/f'trace_{i:03d}.npz', reference_state=trace_a['state'], optimized_state=trace_b['state'],
                            reference_action=trace_a['action'], optimized_action=trace_b['action'], goal=trial['goal'])
    report = {'protocol': protocol, 'export': exported, 'latency': latency,
              'p50_speedup': latency['sb3_predict']['p50_ms']/latency['numpy_actor']['p50_ms'],
              'paired_block_speedup_mean': float(ratios.mean()),
              'paired_block_bootstrap_95pct_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
              'observed_states': len(cases), 'max_action_vector_difference_m': float(agreement.max()),
              'all_success_outcomes_match': all(r['reference_success'] == r['optimized_success'] for r in traces),
              'max_trace_position_vector_difference_mm': max(r['max_tool_and_object_position_vector_difference_mm'] for r in traces),
              'torch_direct_p50_speedup': latency['sb3_predict']['p50_ms']/latency['torch_direct']['p50_ms'],
              'torch_direct_bit_identical_all_actions_and_states': all(r['bit_identical_actions'] and r['bit_identical_states'] for r in exact_traces),
              'traces': traces, 'torch_direct_traces': exact_traces,
              'limitations': ['Local CPU measurements, not edge-hardware measurements',
                             'Float32 arithmetic order changes rounding; not bit-identical',
                             'Optimization targets the standard SB3 prediction wrapper; the controller and simulator still take time',
                             'Thirty timing blocks are repeated measurements on one machine, not independent hardware samples']}
    write_json(args.out/'metrics.json', report)
    print(f'Actor p50 speedup {report["p50_speedup"]:.2f}x; max action difference {agreement.max()*1e6:.4f} micrometres; trajectory checks {report["all_success_outcomes_match"]}', flush=True)


if __name__ == '__main__':
    main()
