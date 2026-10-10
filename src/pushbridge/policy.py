import numpy as np

from .sim import HEIGHT, MAX_ACTION, WIDTH, clip_action


def goal_frame(puck, goal):
    direction = goal-puck
    direction = direction/max(np.linalg.norm(direction), 1e-8)
    return np.stack([direction, [-direction[1], direction[0]]], axis=1)


def geometric_action(state, goal):
    """Explicit hand-coded seed/baseline: approach behind the puck, then push."""
    tool, puck = state[:2], state[2:]
    direction = goal_frame(puck, goal)[:, 0]
    relative = tool-puck
    behind = puck-direction*0.023
    lateral = relative-direction*np.dot(relative, direction)
    if np.dot(relative, direction) > -0.015 or np.linalg.norm(lateral) > 0.005:
        # Route around the puck before crossing to its back side.
        side = np.array([-direction[1], direction[0]])
        sign = 1 if np.dot(relative, side) >= 0 else -1
        if np.dot(relative, direction) > -0.01:
            behind = puck-direction*0.025+side*sign*0.035
    return clip_action(behind-tool if np.linalg.norm(behind-tool) > 0.005 else direction*MAX_ACTION)


class Policy:
    def __init__(self, world, mode='mpc', seed=0, uncertainty_weight=0.5):
        self.world, self.mode = world, mode
        self.rng = np.random.default_rng(seed)
        self.index = 0
        self.uncertainty_weight = uncertainty_weight

    def prior(self, state, goal):
        frame = goal_frame(state[2:], goal)
        relative = (state[:2]-state[2:]) @ frame
        distance = np.linalg.norm(self.world.prior_relative-relative, axis=1)
        indices = np.argsort(distance)[:5]
        weights = 1/np.maximum(distance[indices], 0.001)
        canonical = np.average(self.world.prior_action[indices], axis=0, weights=weights)
        return clip_action(frame @ canonical)

    def act(self, state, goal):
        if self.mode == 'geometry':
            return geometric_action(state, goal)
        if self.mode == 'replay':
            replay = self.world.checkpoint['replay_actions'].numpy()
            action = replay[min(self.index, len(replay)-1)] if self.index < len(replay) else np.zeros(2)
            self.index += 1
            return clip_action(action)
        if self.mode == 'prior':
            return self.prior(state, goal)
        base, geo = self.prior(state, goal), geometric_action(state, goal)
        actions = np.vstack([base, geo, np.zeros(2), base+self.rng.normal(0, 0.003, (24, 2)), geo+self.rng.normal(0, 0.003, (24, 2))])
        actions = actions*np.minimum(1, MAX_ACTION/np.maximum(np.linalg.norm(actions, axis=1, keepdims=True), 1e-8))
        initial_actions = actions.copy()
        tools = np.repeat(state[:2][None], len(actions), axis=0)
        pucks = np.repeat(state[2:][None], len(actions), axis=0)
        uncertainty = np.zeros(len(actions))
        cost = np.zeros(len(actions))
        for h in range(4):
            actions = np.clip(tools+actions, [0.012, 0.012], [WIDTH-0.012, HEIGHT-0.012])-tools
            delta, radius = self.world.predict(tools-pucks, actions)
            tools += actions
            pucks += delta
            uncertainty += radius
            cost += np.linalg.norm(pucks-goal, axis=1)*(1+0.3*h)
            # Subsequent moves use the explicit geometric proposal. First moves
            # include a learned demonstration proposal; the WM ranks all of them.
            actions = np.array([geometric_action(np.r_[t, p], goal) for t, p in zip(tools, pucks)])
        direction = goal-pucks
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
        desired = pucks-direction*0.022
        cost += 0.6*np.linalg.norm(tools-desired, axis=1)
        cost += self.uncertainty_weight*uncertainty
        return initial_actions[np.argmin(cost)]
