"""State-based SAC, with an optional actor initialization from personal videos.

The actor sees the same tool/object/goal positions available to MPC. Personal
human transitions are used only as supervised action labels, never presented
as robot-physics transitions to the critic or its replay buffer.
"""
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .data import load_episodes, sha256, transitions, write_json
from .model import WorldModel
from .sim import HEIGHT, MAX_ACTION, WIDTH, PushEnv, clip_action


def observation(state, goal):
    state, goal = np.asarray(state), np.asarray(goal)
    return np.r_[(state[:2]-state[2:])/.05, (goal-state[2:])/.20].astype(np.float32)


def potential(state, goal):
    direction = goal-state[2:]
    distance = np.linalg.norm(direction)
    behind = state[2:]-.023*direction/max(distance, 1e-8)
    return -100*distance-10*np.linalg.norm(state[:2]-behind)


class PushGym(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, horizon=120):
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-10., 10., shape=(4,), dtype=np.float32)
        self.physics = PushEnv()
        self.horizon = horizon
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        puck = self.np_random.uniform([.075, .065], [.13, .145])
        self.goal = self.np_random.uniform([.21, .06], [.25, .15])
        direction = (self.goal-puck)/np.linalg.norm(self.goal-puck)
        tool = puck-.040*direction+self.np_random.normal(0, .002, 2)
        friction = float(self.np_random.choice([.15, .25, .40]))
        for name in ['floor', 'puck_geom']:
            self.physics.model.geom(name).friction[0] = friction
        self.state = self.physics.reset(puck, tool, self.goal)
        self.count = 0
        return observation(self.state, self.goal), {'friction': friction}

    def step(self, action):
        before = potential(self.state, self.goal)
        self.state = self.physics.step(np.asarray(action)*MAX_ACTION)
        self.count += 1
        success = False
        hold_steps = 0
        if np.linalg.norm(self.state[2:]-self.goal) < .018:
            for _ in range(10):
                self.state = self.physics.step([0, 0])
                hold_steps += 1
            success = bool(np.linalg.norm(self.state[2:]-self.goal) < .018)
        outside = bool(np.any(self.state[2:] < 0) or np.any(self.state[2:] > [WIDTH, HEIGHT]))
        reward = potential(self.state, self.goal)-before-.01+10*success-5*outside
        info = {'is_success': success, 'goal_distance_mm': float(np.linalg.norm(self.state[2:]-self.goal)*1000),
                'hold_control_steps': hold_steps}
        return observation(self.state, self.goal), float(reward), success or outside, self.count >= self.horizon, info

    def close(self):
        self.physics.close()


class SACPolicy:
    def __init__(self, path):
        self.agent = SAC.load(path, device='cpu')

    def act(self, state, goal):
        action, _ = self.agent.predict(observation(state, goal), deterministic=True)
        return clip_action(action*MAX_ACTION)


def demonstration_arrays(dataset, checkpoint, seed):
    manifest, episodes = load_episodes(dataset)
    world = WorldModel.load(checkpoint)
    if sha256(Path(dataset)/'manifest.json') != world.checkpoint['manifest_sha256']:
        raise ValueError('The RL dataset must match the world-model checkpoint manifest')
    train_ids = set(world.checkpoint['train_ids'])
    samples, labels = [], []
    for ep in episodes:
        if ep['id'] not in train_ids:
            continue
        state, action, _, _ = transitions(ep)
        # Ignore almost-stationary marker jitter and already-reached goals.
        good = (np.linalg.norm(action, axis=1) > .0004) & (np.linalg.norm(state[:, 2:]-[.225, .105], axis=1) > .020)
        for s, a in zip(state[good], action[good]):
            samples.append(observation(s, [.225, .105]))
            labels.append(clip_action(a)/MAX_ACTION)
    if len(samples) < 20:
        raise ValueError('Too few training-only moving-finger labels for actor initialization')
    base = len(samples)
    obs, act = np.array(samples), np.array(labels)
    # Rotation augmentation rotates relative state, goal direction and labels
    # together. No additional physics transitions or successful labels are made.
    rng = np.random.default_rng(seed)
    obs, act = np.repeat(obs, 16, axis=0), np.repeat(act, 16, axis=0)
    angle = rng.uniform(-np.pi/3, np.pi/3, len(obs))
    rotation = np.stack([np.stack([np.cos(angle), -np.sin(angle)], 1),
                         np.stack([np.sin(angle), np.cos(angle)], 1)], 1)
    obs = np.c_[np.einsum('nij,nj->ni', rotation, obs[:, :2]), np.einsum('nij,nj->ni', rotation, obs[:, 2:])]
    act = np.einsum('nij,nj->ni', rotation, act)
    return obs.astype(np.float32), act.astype(np.float32), {'training_clip_ids': sorted(train_ids),
            'base_labels': base, 'augmented_labels': len(obs), 'manifest_sha256': sha256(Path(dataset)/'manifest.json'),
            'personal_source': manifest['source'], 'source_coordinate_units': 'normalised simulator metres'}


def initialize_actor(agent, obs, actions, updates=500, seed=0, learning_rate=.001):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=learning_rate)
    losses = []
    for _ in range(updates):
        index = rng.integers(len(obs), size=128)
        x, y = torch.tensor(obs[index]), torch.tensor(actions[index])
        loss = (agent.actor(x, deterministic=True)-y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {'updates': updates, 'batch_size': 128, 'learning_rate': learning_rate,
            'last_50_updates_action_mse': float(np.mean(losses[-50:]))}


def validation_check(agent, episodes=10, seed=9012):
    from .evaluation import make_trials
    env = PushEnv()
    outcomes = []
    try:
        for trial in make_trials(episodes, seed):
            for name in ['floor', 'puck_geom']:
                env.model.geom(name).friction[0] = trial['friction']
            state = env.reset(trial['puck'], trial['tool'], trial['goal'])
            success = False
            for _ in range(120):
                action, _ = agent.predict(observation(state, trial['goal']), deterministic=True)
                state = env.step(action*MAX_ACTION)
                if np.linalg.norm(state[2:]-trial['goal']) < .018:
                    for _ in range(10):
                        state = env.step([0, 0])
                    success = bool(np.linalg.norm(state[2:]-trial['goal']) < .018)
                    if success:
                        break
                if np.any(state[2:] < 0) or np.any(state[2:] > [WIDTH, HEIGHT]):
                    break
            outcomes.append({'success': success, 'final_distance_mm': float(np.linalg.norm(state[2:]-trial['goal'])*1000)})
    finally:
        env.close()
    return {'successes': sum(o['success'] for o in outcomes), 'episodes': episodes,
            'mean_final_distance_mm': float(np.mean([o['final_distance_mm'] for o in outcomes]))}


class LearningCurve(BaseCallback):
    def __init__(self, out, interval, started):
        super().__init__()
        self.out, self.interval, self.started = Path(out), interval, started
        self.rows = []
        self.hold_steps = 0

    def measure(self, steps):
        result = validation_check(self.model)
        row = {'sim_interaction_steps': steps, 'terminal_hold_control_steps': self.hold_steps,
               'elapsed_seconds_including_validation': time.perf_counter()-self.started, **result}
        self.rows.append(row)
        write_json(self.out/'learning_curve.json', {'validation_seed': 9012, 'selection': 'Final fixed-budget model; curves are not used to select checkpoints.', 'rows': self.rows})
        print(f'{self.out.name}: {steps} sim steps | validation {result["successes"]}/{result["episodes"]} | {result["mean_final_distance_mm"]:.1f} mm', flush=True)

    def _on_training_start(self):
        self.measure(0)

    def _on_step(self):
        self.hold_steps += sum(info.get('hold_control_steps', 0) for info in self.locals['infos'])
        if self.num_timesteps % self.interval == 0:
            self.measure(self.num_timesteps)
            self.model.save(self.out/'latest')
        return True


def train_sac(dataset, checkpoint, out, steps=40000, seed=0, demo_init=False, eval_interval=5000):
    if steps < 1000 or eval_interval < 1000:
        raise ValueError('Use at least 1000 interaction steps and validation interval')
    out = Path(out)
    if (out/'sac.zip').exists():
        raise ValueError('RL checkpoint already exists; choose a new output folder')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    env = PushGym()
    parameters = {'learning_rate': .0003, 'buffer_size': 100000, 'learning_starts': 1000,
                  'batch_size': 128, 'tau': .005, 'gamma': .99, 'train_freq': 4,
                  'gradient_steps': 1, 'ent_coef': 'auto_0.1'}
    agent = SAC('MlpPolicy', env, **parameters, policy_kwargs={'net_arch': [64, 64]}, device='cpu', seed=seed, verbose=0)
    # Same initial exploration variance for both variants; BC only changes mean.
    torch.nn.init.zeros_(agent.actor.log_std.weight)
    torch.nn.init.constant_(agent.actor.log_std.bias, -1.)
    initialization = None
    if demo_init:
        obs, action, data_info = demonstration_arrays(dataset, checkpoint, seed)
        initialization = {**data_info, **initialize_actor(agent, obs, action, seed=seed)}
        agent.save(out/'actor_initialized')
    np.random.seed(seed)
    torch.manual_seed(seed)
    callback = LearningCurve(out, eval_interval, started)
    try:
        agent.learn(total_timesteps=steps, callback=callback)
        agent.save(out/'sac')
    finally:
        env.close()
    report = {'algorithm': 'SAC', 'implementation': 'Stable Baselines3 2.9.0', 'seed': seed,
              'variant': 'personal-demo actor initialization + SAC' if demo_init else 'SAC from scratch',
              'sim_interaction_steps': int(agent.num_timesteps), 'terminal_hold_control_steps': callback.hold_steps,
              'total_training_control_steps': int(agent.num_timesteps)+callback.hold_steps,
              'training_seconds_including_validation': time.perf_counter()-started,
              'reward': 'Delta of (-100 * object-goal distance - 10 * tool-to-behind-object distance) - 0.01; +10 stable success; -5 outside workspace.',
              'observation': '4 floats: (tool-object)/0.05 and (goal-object)/0.20. MuJoCo ground truth; no images/language/velocities.',
              'action': 'Two [-1,1] values scaled by 8 mm and norm-clipped; joint servos realize each move over 0.1 s.',
              'episode_horizon': 120, 'success': 'Within 18 mm after a one-second zero-action hold.',
              'training_distribution': 'Same ranges of starts, goals and friction as benchmark; separate random sequences.',
              'parameters': parameters, 'network': [64, 64], 'initial_log_std': -1., 'demo_initialization': initialization,
              'model_selection': 'Final checkpoint after the fixed simulation budget, no test selection.',
              'limitations': ['State-based RL, not a VLA.', 'Dense hand-designed geometric reward.', 'Demo labels have human-to-robot contact and tracking differences.',
                              'Imitation initialization changes actor means only; critics and replay start from simulator data.']}
    write_json(out/'training_metrics.json', report)
    return out/'sac.zip'


def experiment(dataset, checkpoint, out, steps=40000, seeds=(0, 1, 2), eval_interval=5000):
    """Fixed-budget paired ablation, followed by one shared final benchmark."""
    out = Path(out)
    if (out/'protocol.json').exists():
        raise ValueError('Experiment folder already exists; choose a new output folder')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'protocol.json', {'algorithm': 'SAC', 'variants': ['scratch', 'demo_initialized'],
               'training_seeds': list(seeds), 'interaction_steps_per_model': steps,
               'validation_seed': 9012, 'validation_episodes': 10, 'validation_interval': eval_interval,
               'test_seed': 2026, 'test_episodes_per_model': 20, 'test_horizon': 120,
               'selection': 'Final fixed-budget checkpoint; validation curves do not select the model.',
               'comparison': 'Same simulator-state observations, action bounds, physical environment and paired trials as MPC/geometry.',
               'personal_data_role': 'Only world-model training clips initialize actor means. No human transitions enter robot replay or critics.',
               'additional_baseline': 'Imitation-only initialized actor, before simulator learning, for every training seed.',
               'resource_note': 'RL receives additional simulator interactions and a dense engineered reward; MPC receives no simulation training.',
               'world_model_checkpoint_sha256': sha256(checkpoint), 'dataset_manifest_sha256': sha256(Path(dataset)/'manifest.json')})
    models = {}
    for seed in seeds:
        for variant, demo in [('scratch', False), ('demo_initialized', True)]:
            name = f'sac_{variant}_seed{seed}'
            models[name] = train_sac(dataset, checkpoint, out/name, steps, seed, demo, eval_interval)
            if demo:
                models[f'bc_only_seed{seed}'] = out/name/'actor_initialized.zip'
    from .evaluation import evaluate
    result = evaluate(checkpoint, out/'evaluation', episodes=20, steps=120, seed=2026, rl_models=models)
    write_json(out/'model_paths.json', {name: str(path.relative_to(out)) for name, path in models.items()})
    return result
