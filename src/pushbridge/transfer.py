"""Frozen-policy transfer and fixed-budget post-training on Compact 3R."""
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .data import sha256, write_json
from .embodiment import COMPACT_SPEC, CONTACT_SCALE, CompactGeometry, CompactPushEnv, ContactAdapter, canonical_state
from .evaluation import make_trials, wilson
from .policy import Policy
from .rl import PushGym, SACPolicy, observation
from .sim import MAX_ACTION, WIDTH, HEIGHT
from .video import VideoWriter


def compact_potential(state, goal):
    direction = goal-state[2:]
    distance = np.linalg.norm(direction)
    behind = state[2:]-.018*direction/max(distance, 1e-8)
    return -100*distance-10*np.linalg.norm(state[:2]-behind)


class CompactGym(PushGym):
    def __init__(self, horizon=120):
        super().__init__(horizon)
        self.physics.close()
        self.physics = CompactPushEnv()

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        return observation(canonical_state(self.state), self.goal), info

    def step(self, action):
        before = compact_potential(self.state, self.goal)
        self.state = self.physics.step(np.asarray(action)*MAX_ACTION/CONTACT_SCALE)
        self.count += 1
        success, hold_steps = False, 0
        if np.linalg.norm(self.state[2:]-self.goal) < .018:
            for _ in range(10):
                self.state = self.physics.step([0, 0])
                hold_steps += 1
            success = bool(np.linalg.norm(self.state[2:]-self.goal) < .018)
        outside = bool(np.any(self.state[2:] < 0) or np.any(self.state[2:] > [WIDTH, HEIGHT]))
        reward = compact_potential(self.state, self.goal)-before-.01+10*success-5*outside
        info = {'is_success': success, 'goal_distance_mm': float(np.linalg.norm(self.state[2:]-self.goal)*1000),
                'hold_control_steps': hold_steps}
        return observation(canonical_state(self.state), self.goal), float(reward), success or outside, self.count >= self.horizon, info


def run_trial(policy, trial, env_class=CompactPushEnv, horizon=120, writer=None, label=''):
    env = env_class(trial['friction'])
    state = env.reset(trial['puck'], trial['tool'], trial['goal'])
    history, actions, joints, times = [state.copy()], [], [env.data.qpos[:env.model.nu].copy()], []
    success = False
    def frame():
        img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (0, 0), (960, 70), (20, 27, 35), -1)
        cv2.putText(img, label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 240, 240), 2)
        cv2.putText(img, f'Physical contact | goal error {1000*np.linalg.norm(state[2:]-trial["goal"]):.1f} mm', (18, 55), cv2.FONT_HERSHEY_SIMPLEX, .47, (110, 220, 200), 1)
        return img
    def advance(a):
        nonlocal state
        state = env.step(a)
        history.append(state.copy())
        actions.append(np.asarray(a).copy())
        joints.append(env.data.qpos[:env.model.nu].copy())
        if writer:
            writer.write(frame())
    try:
        for step in range(horizon):
            tick = time.perf_counter()
            a = policy.act(state, trial['goal'])
            times.append((time.perf_counter()-tick)*1000)
            advance(a)
            if np.linalg.norm(state[2:]-trial['goal']) < .018:
                for _ in range(10):
                    advance(np.zeros(2))
                success = bool(np.linalg.norm(state[2:]-trial['goal']) < .018)
                if success:
                    break
            if np.any(state[2:] < 0) or np.any(state[2:] > [WIDTH, HEIGHT]):
                break
        poster = frame() if writer else None
        diagnostics = env.adapter_diagnostics.copy() if hasattr(env, 'adapter_diagnostics') else {}
    finally:
        env.close()
    row = {'success': success, 'steps': step+1, 'total_control_steps': len(actions),
           'final_distance_mm': float(np.linalg.norm(state[2:]-trial['goal'])*1000),
           'initial_puck': trial['puck'].tolist(), 'initial_tool': trial['tool'].tolist(),
           'goal': trial['goal'].tolist(), 'friction': trial['friction'], 'adapter': diagnostics}
    return row, {'state': np.array(history), 'action': np.array(actions), 'joint': np.array(joints),
                 'goal': trial['goal'], 'decision_ms': np.array(times)}, poster


def validate_transfer(agent, episodes=10):
    class Actor:
        def act(self, state, goal):
            a, _ = agent.predict(observation(state, goal), deterministic=True)
            return a*MAX_ACTION
    rows = [run_trial(ContactAdapter(Actor()), t)[0] for t in make_trials(episodes, 7131)]
    return {'successes': sum(r['success'] for r in rows), 'episodes': episodes,
            'mean_final_distance_mm': float(np.mean([r['final_distance_mm'] for r in rows]))}


class TransferCurve(BaseCallback):
    def __init__(self, out, started, interval=5000):
        super().__init__()
        self.out, self.started, self.interval = Path(out), started, interval
        self.rows, self.hold_steps = [], 0

    def measure(self, steps):
        row = {'adaptation_steps': steps, 'terminal_hold_control_steps': self.hold_steps,
               'elapsed_seconds_including_validation': time.perf_counter()-self.started, **validate_transfer(self.model)}
        self.rows.append(row)
        write_json(self.out/'learning_curve.json', {'seed': 7131, 'rows': self.rows, 'model_selection': 'Final fixed-budget checkpoint'})
        print(f'{self.out.name}: {steps} adaptation steps | validation {row["successes"]}/10', flush=True)

    def _on_training_start(self):
        self.measure(0)

    def _on_step(self):
        self.hold_steps += sum(info.get('hold_control_steps', 0) for info in self.locals['infos'])
        if self.num_timesteps % self.interval == 0:
            self.measure(self.num_timesteps)
        return True


def train_transfer(source, out, steps=20000, seed=0):
    out = Path(out)
    if (out/'sac.zip').exists():
        raise ValueError('Transfer checkpoint exists; use a new output folder')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    env = CompactGym()
    # Source checkpoints contain parameters, not replay data. The loaded
    # actor/critics transfer; the target replay starts empty and contains only
    # compact-arm transitions. Source simulation budget is reported separately.
    agent = SAC.load(source, env=env, device='cpu', seed=seed, learning_rate=.00015)
    agent.set_random_seed(seed)
    callback = TransferCurve(out, started)
    try:
        agent.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=True)
        agent.save(out/'sac')
    finally:
        env.close()
    write_json(out/'training_metrics.json', {
        'seed': seed, 'source_checkpoint_sha256': sha256(source), 'source_sim_interactions': 40000,
        'target_sim_interactions': steps, 'terminal_hold_control_steps': callback.hold_steps,
        'elapsed_seconds_including_validation': time.perf_counter()-started, 'learning_rate': .00015,
        'transferred': 'Actor, critics, entropy parameter and optimizer states; target replay initially empty',
        'personal_data_role': 'Source actor initialized from the three personal training clips, then source SAC, then target SAC',
        'target_reward': 'Potential progress using 18 mm behind-object pose; -0.01 per decision; +10 stable success; -5 outside',
        'observation': 'Contact-normalised relative tool/object coordinates and goal displacement; exact simulator states',
        'model_selection': 'Final checkpoint after fixed budget; validation seed 7131; test seed 2027',
        'embodiment': COMPACT_SPEC,
    })
    return out/'sac.zip'


def transfer_experiment(source_root, checkpoint, out, steps=20000, seeds=(0, 1, 2), episodes=20):
    from .model import WorldModel
    out, source_root = Path(out), Path(source_root)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'embodiment': COMPACT_SPEC, 'contact_scale': CONTACT_SCALE, 'train_seeds': list(seeds),
                'adaptation_steps_per_model': steps, 'source_steps_per_model': 40000,
                'validation_seed': 7131, 'test_seed': 2027, 'test_episodes': episodes, 'horizon': 120,
                'success_threshold_mm': 18, 'success_hold_s': 1,
                'selection': 'Final fixed-budget checkpoints; no test-based selection',
                'baselines': ['absolute replay', 'geometry', 'contact-adapted geometry', 'contact-adapted MPC',
                              'source SAC direct', 'source SAC contact adapter', 'post-trained SAC contact adapter',
                              'compact-aware classical feedback (follow-up baseline)'],
                'followup_baseline_note': 'CompactGeometry was implemented after the initial transfer comparison, inspected on validation seed 7131, and retained as an additional baseline; original baselines/results are preserved.',
                'limitations': ['Compact planar 3R, not a humanoid or full 3D manipulator',
                                'No vision/language inputs to the policy', 'Only three training seeds and twenty paired task cases',
                                'Post-training receives extra simulator data; frozen transfer has no target training',
                                'Known tool radius is engineered embodiment information']}
    write_json(out/'protocol.json', protocol)
    adapted = {}
    for seed in seeds:
        source = source_root/f'sac_demo_initialized_seed{seed}/sac.zip'
        adapted[seed] = train_transfer(source, out/f'posttrained_seed{seed}', steps, seed)
    world = WorldModel.load(checkpoint)
    factories = {
        'replay': lambda i: Policy(world, 'replay'),
        'geometry': lambda i: Policy(world, 'geometry'),
        'geometry_contact': lambda i: ContactAdapter(Policy(world, 'geometry')),
        'mpc_contact': lambda i: ContactAdapter(Policy(world, 'mpc', 2027+i)),
        'geometry_compact': lambda i: CompactGeometry(),
    }
    for seed in seeds:
        source = source_root/f'sac_demo_initialized_seed{seed}/sac.zip'
        raw, post = SACPolicy(source), SACPolicy(adapted[seed])
        factories[f'source_direct_seed{seed}'] = lambda i, p=raw: p
        factories[f'source_contact_seed{seed}'] = lambda i, p=raw: ContactAdapter(p)
        factories[f'posttrained_seed{seed}'] = lambda i, p=post: ContactAdapter(p)
    trials = make_trials(episodes, 2027)
    results = {}
    for name, factory in factories.items():
        rows, timing, residuals = [], [], []
        for i, trial in enumerate(trials):
            writer = VideoWriter(out/f'{name}.mp4', 10, (960, 640)) if i == 0 else None
            try:
                row, trace, poster = run_trial(factory(i), trial, writer=writer, label=f'Compact 3R | {name}')
            finally:
                if writer:
                    writer.release()
            row['trial'] = i
            rows.append(row)
            timing.extend(trace['decision_ms'])
            residuals.append(row['adapter']['sum_tracking_residual_m']/max(row['adapter']['control_steps'], 1)*1000)
            np.savez_compressed(out/f'{name}_{i:03d}.npz', **trace)
            if poster is not None:
                cv2.imwrite(str(out/f'{name}.jpg'), poster)
        n = sum(r['success'] for r in rows)
        results[name] = {'successes': n, 'episodes': episodes, 'wilson_95pct': wilson(n, episodes),
                         'mean_final_distance_mm': float(np.mean([r['final_distance_mm'] for r in rows])),
                         'p95_decision_ms': float(np.quantile(timing, .95)),
                         'mean_target_tracking_residual_mm': float(np.mean(residuals)), 'runs': rows}
        print(f'Compact {name}: {n}/{episodes} success, {results[name]["mean_final_distance_mm"]:.1f} mm', flush=True)
    write_json(out/'evaluation.json', {'protocol': protocol, 'results': results})
    return results
