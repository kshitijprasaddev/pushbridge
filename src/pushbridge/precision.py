"""An explicit hybrid controller and a stricter, whole-object benchmark.

This is a follow-up experiment. Existing SAC checkpoints were trained to an
18 mm centre-distance criterion. Their weights and original scores stay intact.
The feedback controller uses known geometry and simulator state, not training.
"""
import numpy as np

from .embodiment import CompactPushEnv
from .sim import HEIGHT, WIDTH, clip_action


GOAL_RADIUS = .020
PUCK_RADIUS = .012
FIT_DISTANCE = GOAL_RADIUS - PUCK_RADIUS
HANDOFF_DISTANCE = .030
HOLD_STEPS = 10


def clearance(state, goal):
    """Positive exactly when the entire circular footprint fits inside the goal."""
    return FIT_DISTANCE - float(np.linalg.norm(np.asarray(state)[2:] - goal))


class PrecisionFeedback:
    """Align behind the puck, taper the push, then retract to release contact.

    The stop point is 3 mm from the goal centre, leaving 5 mm of geometric
    margin before the 8 mm containment boundary. All movement is physical.
    """
    def __init__(self):
        self.phase = 'align'
        self.release_remaining = None
        self.last_direction = np.array([1., 0.])

    @property
    def settled(self):
        return self.release_remaining == 0

    def act(self, state, goal):
        tool, puck = state[:2], state[2:]
        error = goal - puck
        distance = float(np.linalg.norm(error))
        if self.release_remaining is not None:
            if self.release_remaining:
                self.phase = 'release'
                self.release_remaining -= 1
                return -self.last_direction * .003
            self.phase = 'hold'
            return np.zeros(2)
        direction = error / max(distance, 1e-8)
        if distance <= .003:
            self.release_remaining = 4
            return self.act(state, goal)
        self.last_direction = direction
        relative = tool - puck
        along = float(relative @ direction)
        lateral = relative - direction * along
        if along > -.010:
            self.phase = 'align'
            side = np.array([-direction[1], direction[0]])
            sign = 1 if relative @ side >= 0 else -1
            return clip_action(puck - direction*.020 + side*sign*.025 - tool, .003)
        if np.linalg.norm(lateral) > .0015 or along < -.019:
            self.phase = 'align'
            return clip_action(puck - direction*.017 - tool, .003)
        self.phase = 'push'
        speed = min(.003, max(.0004, distance*.22))
        return clip_action(direction*speed - .8*lateral, .003)


class PrecisionHandoff:
    """Frozen learned approach followed by explicitly labelled feedback."""
    def __init__(self, policy):
        self.policy = policy
        self.feedback = PrecisionFeedback()
        self.active = False
        self.phase = 'policy'
        self.handoff_step = None
        self.decisions = 0

    @property
    def settled(self):
        return self.active and self.feedback.settled

    def act(self, state, goal):
        if not self.active and np.linalg.norm(state[2:] - goal) <= HANDOFF_DISTANCE:
            self.active = True
            self.handoff_step = self.decisions
        self.decisions += 1
        if self.active:
            action = self.feedback.act(state, goal)
            self.phase = self.feedback.phase
            return action
        return self.policy.act(state, goal)


def run_precision(policy, trial, horizon=120, observer=None):
    """Require containment throughout all ten stationary hold ticks.

    Decision budget matches the old benchmark. As before, verification hold
    ticks are additional physics ticks and are reported separately.
    Observer receives actual MuJoCo data; it cannot influence physics.
    """
    env = CompactPushEnv(trial['friction'])
    state = env.reset(trial['puck'], trial['tool'], trial['goal'])
    states, actions, joints, phases = [state.copy()], [], [env.data.qpos[:3].copy()], ['start']
    success, hold_ticks, stable_hold_max_mm = False, 0, None

    def record(action, phase):
        nonlocal state
        state = env.step(action)
        states.append(state.copy())
        actions.append(np.asarray(action).copy())
        joints.append(env.data.qpos[:3].copy())
        phases.append(phase)
        if observer:
            observer(env, state, phase, len(actions))

    try:
        if observer:
            observer(env, state, 'start', 0)
        for step in range(horizon):
            action = policy.act(state, trial['goal'])
            record(action, getattr(policy, 'phase', 'policy'))
            if clearance(state, trial['goal']) >= 0 and getattr(policy, 'settled', True):
                hold_distances = [np.linalg.norm(state[2:] - trial['goal'])*1000]
                for _ in range(HOLD_STEPS):
                    record(np.zeros(2), 'verify')
                    hold_ticks += 1
                    hold_distances.append(np.linalg.norm(state[2:] - trial['goal'])*1000)
                stable_hold_max_mm = float(max(hold_distances))
                success = stable_hold_max_mm <= FIT_DISTANCE*1000
                if success:
                    break
            if np.any(state[2:] < 0) or np.any(state[2:] > [WIDTH, HEIGHT]):
                break
        row = {
            'success': success, 'decisions': step+1, 'verification_hold_ticks': hold_ticks,
            'total_control_steps': len(actions),
            'final_distance_mm': float(np.linalg.norm(state[2:] - trial['goal'])*1000),
            'final_clearance_mm': clearance(state, trial['goal'])*1000,
            'hold_max_distance_mm': stable_hold_max_mm,
            'handoff_step': getattr(policy, 'handoff_step', None),
            'friction': trial['friction'], 'goal': trial['goal'].tolist(),
            'initial_puck': trial['puck'].tolist(), 'initial_tool': trial['tool'].tolist(),
            'adapter': env.adapter_diagnostics.copy(),
        }
    finally:
        env.close()
    trace = {'state': np.array(states), 'action': np.array(actions), 'joint': np.array(joints),
             'phase': np.array(phases), 'goal': trial['goal']}
    return row, trace
