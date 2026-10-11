"""A strict training objective, separate from the frozen 18 mm experiment.

The action, observation, reset distribution and dense potential stay identical
to CompactGym. Only the terminal success predicate changes: the circular puck
must remain entirely inside the goal during ten zero-action physics ticks.
The verification stop is a benchmark wrapper, not a learned release action.
"""
import numpy as np

from .embodiment import CONTACT_SCALE, canonical_state
from .precision import FIT_DISTANCE, HOLD_STEPS
from .rl import observation
from .sim import HEIGHT, MAX_ACTION, WIDTH
from .transfer import CompactGym, compact_potential


def strict_hold_succeeded(distances):
    """The entry state and every hold state must satisfy containment."""
    values = np.asarray(distances, dtype=float)
    return bool(len(values) == HOLD_STEPS + 1 and np.isfinite(values).all()
                and np.all(values <= FIT_DISTANCE))


class StrictCompactGym(CompactGym):
    """SAC receives a +10 bonus only for the strict benchmark objective."""

    def step(self, action):
        before = compact_potential(self.state, self.goal)
        self.state = self.physics.step(np.asarray(action) * MAX_ACTION / CONTACT_SCALE)
        self.count += 1
        distances = [float(np.linalg.norm(self.state[2:] - self.goal))]
        hold_steps = 0
        if distances[0] <= FIT_DISTANCE:
            for _ in range(HOLD_STEPS):
                self.state = self.physics.step(np.zeros(2))
                hold_steps += 1
                distances.append(float(np.linalg.norm(self.state[2:] - self.goal)))
        success = strict_hold_succeeded(distances)
        outside = bool(np.any(self.state[2:] < 0) or np.any(self.state[2:] > [WIDTH, HEIGHT]))
        reward = compact_potential(self.state, self.goal) - before - .01 + 10 * success - 5 * outside
        info = {'is_success': success, 'goal_distance_mm': distances[-1] * 1000,
                'hold_control_steps': hold_steps,
                'hold_max_distance_mm': max(distances) * 1000 if hold_steps else None}
        return (observation(canonical_state(self.state), self.goal), float(reward),
                success or outside, self.count >= self.horizon, info)
