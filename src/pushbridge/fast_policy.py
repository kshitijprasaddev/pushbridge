"""A deterministic SAC actor exported to three NumPy matrix multiplies.

This preserves trained float32 weights and the original observation/action
normalisation. It removes training, critic and distribution machinery from
inference; it does not quantise or retrain the policy.
"""
from pathlib import Path

import numpy as np

from .sim import MAX_ACTION, clip_action


class FastActor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.weights = [(np.ascontiguousarray(d[f'w{i}']), d[f'b{i}'].copy()) for i in range(3)]

    def predict(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for w, b in self.weights[:-1]:
            x = np.maximum(x@w+b, np.float32(0))
        w, b = self.weights[-1]
        output = np.tanh(x@w+b)
        # Match SB3's unscale_action even for a symmetric [-1, 1] Box.
        return -1.+.5*(output+1.)*2.

    def act(self, state, goal):
        s, g = np.asarray(state), np.asarray(goal)
        obs = np.r_[(s[:2]-s[2:])/.05, (g-s[2:])/.20].astype(np.float32)
        return clip_action(self.predict(obs)*MAX_ACTION)


class DirectTorchActor:
    """Skip SB3 prediction/distribution wrappers with the same Torch kernels."""
    def __init__(self, agent):
        if isinstance(agent, (str, Path)):
            import torch
            from types import SimpleNamespace
            with np.load(agent, allow_pickle=False) as d:
                layers = []
                for i in range(3):
                    w, b = d[f'w{i}'], d[f'b{i}']
                    linear = torch.nn.Linear(w.shape[0], w.shape[1])
                    with torch.no_grad():
                        linear.weight.copy_(torch.from_numpy(w.T.copy()))
                        linear.bias.copy_(torch.from_numpy(b.copy()))
                    layers.append(linear.eval())
                    if i < 2:
                        layers.append(torch.nn.ReLU())
            self.actor = SimpleNamespace(latent_pi=torch.nn.Sequential(*layers[:-1]).eval(), mu=layers[-1])
        else:
            self.actor = agent.actor

    def act(self, state, goal):
        import torch
        s, g = np.asarray(state), np.asarray(goal)
        obs = np.r_[(s[:2]-s[2:])/.05, (g-s[2:])/.20].astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(obs.reshape(1, 4))
            output = torch.tanh(self.actor.mu(self.actor.latent_pi(x))).cpu().numpy()[0]
        output = -1.+.5*(output+1.)*2.
        return clip_action(output*MAX_ACTION)


def export_actor(agent, path):
    import torch
    layers = [layer for layer in agent.actor.latent_pi if isinstance(layer, torch.nn.Linear)]
    other = [layer for layer in agent.actor.latent_pi if not isinstance(layer, (torch.nn.Linear, torch.nn.ReLU))]
    if len(layers) != 2 or other:
        raise ValueError('This export supports the tested two-layer ReLU SAC actor only')
    layers.append(agent.actor.mu)
    weights = {}
    for i, layer in enumerate(layers):
        weights[f'w{i}'] = np.ascontiguousarray(layer.weight.detach().cpu().numpy().T)
        weights[f'b{i}'] = layer.bias.detach().cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **weights)
    return {'layers': [4, 64, 64, 2], 'precision': 'float32', 'parameters': sum(v.size for v in weights.values()),
            'method': 'Export deterministic mean actor; two ReLU layers and tanh output; remove training/distribution wrappers',
            'file_bytes': Path(path).stat().st_size}
