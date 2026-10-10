"""Rotation-equivariant, contact-gated state dynamics learned from video tracks.

This is a small state world model, not a video generator or a VLA. Geometric
structure is an explicit prior; contact range and displacement gains are learned.
"""
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import episode_split, load_episodes, sha256, transitions, write_json


class ContactDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 2))
        self.range_logit = nn.Parameter(torch.tensor(-0.9))

    def forward(self, relative, action):
        distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        magnitude = action.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        normal = -relative/distance
        cosine = (normal*action).sum(-1, keepdim=True)/magnitude
        fraction = (-(relative*action).sum(-1, keepdim=True)/(magnitude*magnitude)).clamp(0, 1)
        swept_distance = (relative+fraction*action).norm(dim=-1, keepdim=True)
        feature = torch.cat([distance/0.03, magnitude/0.01, cosine, swept_distance/0.03], -1)
        gain = torch.sigmoid(self.net(feature))
        contact_range = 0.008 + 0.037*torch.sigmoid(self.range_logit)
        gate = torch.sigmoid((contact_range-swept_distance)/0.0025) * torch.sigmoid(12*(cosine+0.05))
        radial = (action*normal).sum(-1, keepdim=True).clamp_min(0)*normal
        return gate*(gain[:, :1]*action + gain[:, 1:]*radial)


class WorldModel:
    def __init__(self, models, checkpoint):
        self.models = models
        self.checkpoint = checkpoint
        self.prior_relative = checkpoint['prior_relative'].numpy()
        self.prior_action = checkpoint['prior_action'].numpy()
        self.conformal_scale = float(checkpoint.get('conformal_scale', 1))

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        models = []
        for weights in payload['models']:
            model = ContactDynamics()
            model.load_state_dict(weights)
            model.eval()
            models.append(model)
        return cls(models, payload)

    @torch.no_grad()
    def predict(self, relative, action):
        relative = torch.as_tensor(np.asarray(relative), dtype=torch.float32).reshape(-1, 2)
        action = torch.as_tensor(np.asarray(action), dtype=torch.float32).reshape(-1, 2)
        predictions = torch.stack([m(relative, action) for m in self.models])
        mean = predictions.mean(0).numpy()
        # Ensembles can agree and still be wrong. This is calibrated on validation clips.
        scale = np.maximum(predictions.std(0, unbiased=False).norm(dim=-1).numpy(), 0.0005)
        return mean, scale*self.conformal_scale


def arrays(episodes):
    triples = [transitions(e)[:3] for e in episodes]
    state, action, delta = [np.concatenate([t[i] for t in triples]) for i in range(3)]
    return state, action, delta


def canonical_prior(episodes):
    rels, acts = [], []
    for ep in episodes:
        s, a, _, _ = transitions(ep)
        final = ep['state'][np.flatnonzero(ep['valid'])[-1], 2:]
        direction = final-s[:, 2:]
        norm = np.linalg.norm(direction, axis=1)
        good = (norm > 0.015) & (np.linalg.norm(a, axis=1) > 0.0004)
        d = direction[good]/norm[good, None]
        side = np.stack([-d[:, 1], d[:, 0]], axis=1)
        r = s[good, :2]-s[good, 2:]
        rels.append(np.stack([np.sum(r*d, 1), np.sum(r*side, 1)], 1))
        acts.append(np.stack([np.sum(a[good]*d, 1), np.sum(a[good]*side, 1)], 1))
    if not rels or sum(len(r) for r in rels) < 10:
        raise ValueError('Too few moving finger samples to construct a demonstration prior')
    return np.concatenate(rels), np.concatenate(acts)


def analytic_delta(relative, action):
    distance = np.linalg.norm(relative, axis=1, keepdims=True)
    normal = -relative/np.maximum(distance, 1e-8)
    fraction = np.clip(-np.sum(relative*action, 1, keepdims=True)/np.maximum(np.sum(action*action, 1, keepdims=True), 1e-10), 0, 1)
    contact = (np.linalg.norm(relative+fraction*action, axis=1, keepdims=True) < 0.021)
    return contact*np.maximum(np.sum(normal*action, 1, keepdims=True), 0)*normal


def prediction_report(world, episodes):
    state, action, delta = arrays(episodes)
    prediction, radius = world.predict(state[:, :2]-state[:, 2:], action)
    errors = np.linalg.norm(prediction-delta, axis=1)
    moving = np.linalg.norm(delta, axis=1) > 0.001
    result = {'transitions': len(state), 'moving_transitions': int(moving.sum()),
              'mean_error_mm': float(errors.mean()*1000), 'p90_error_mm': float(np.quantile(errors, 0.9)*1000),
              'zero_motion_error_mm': float(np.linalg.norm(delta, axis=1).mean()*1000),
              'analytic_contact_error_mm': float(np.linalg.norm(analytic_delta(state[:, :2]-state[:, 2:], action)-delta, axis=1).mean()*1000),
              'moving_mean_error_mm': float(errors[moving].mean()*1000) if moving.any() else None,
              'validation_scaled_radius_coverage': float(np.mean(errors <= radius)),
              'mean_prediction_radius_mm': float(radius.mean()*1000)}
    for horizon in [5, 10]:
        rollout_errors = []
        for ep in episodes:
            s, t, v = ep['state'], ep['time'], ep['valid']
            accepted = np.zeros(len(s)-1, dtype=bool)
            accepted[transitions(ep)[3]] = True
            for start in range(0, len(s)-horizon, horizon):
                if not accepted[start:start+horizon].all():
                    continue
                predicted_object = s[start, 2:].copy()
                for k in range(start, start+horizon):
                    predicted_object += world.predict(s[k, :2]-predicted_object, s[k+1, :2]-s[k, :2])[0][0]
                rollout_errors.append(np.linalg.norm(predicted_object-s[start+horizon, 2:]))
        result[f'{horizon}_step_open_loop_error_mm'] = float(np.mean(rollout_errors)*1000) if rollout_errors else None
        result[f'{horizon}_step_windows'] = len(rollout_errors)
    return result


def train(dataset, out, epochs=200, seed=7, ensemble=3):
    if epochs < 1 or ensemble < 2:
        raise ValueError('epochs must be positive; ensemble must be at least 2')
    torch.set_num_threads(2)
    manifest, episodes = load_episodes(dataset)
    train_eps, validation_eps, test_eps = episode_split(episodes, seed)
    state, action, delta = arrays(train_eps)
    if np.sum(np.linalg.norm(delta, axis=1) > 0.001) < 20:
        raise ValueError('Training split contains fewer than 20 object-moving transitions. Record longer, varied pushes.')
    models, losses = [], []
    started = time.perf_counter()
    # Bootstrap entire clips, rather than pretending adjacent frames are independent.
    for member in range(ensemble):
        torch.manual_seed(seed+member)
        rng = np.random.default_rng(seed+member)
        boot = [train_eps[i] for i in rng.integers(len(train_eps), size=len(train_eps))]
        bs, ba, by = arrays(boot)
        br = torch.tensor(bs[:, :2]-bs[:, 2:], dtype=torch.float32)
        ba, by = torch.tensor(ba, dtype=torch.float32), torch.tensor(by, dtype=torch.float32)
        model = ContactDynamics()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
        best_loss, best_weights = float('inf'), None
        vs, va, vy = arrays(validation_eps)
        vr = torch.tensor(vs[:, :2]-vs[:, 2:], dtype=torch.float32)
        va, vy = torch.tensor(va, dtype=torch.float32), torch.tensor(vy, dtype=torch.float32)
        for epoch in range(epochs):
            model.train()
            # Give movement transitions extra weight so stationary video frames cannot dominate.
            weights = 1 + 3*(by.norm(dim=-1) > 0.001).float()
            loss = (((model(br, ba)-by)/0.01).square().mean(-1)*weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float((model(vr, va)-vy).square().mean())
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_weights)
        models.append(model)
        losses.append(best_loss)
        print(f'Ensemble {member+1}/{ensemble}: best validation coordinate RMSE {1000*np.sqrt(best_loss):.2f} mm')
    prior_r, prior_a = canonical_prior(train_eps)
    longest = max(train_eps, key=lambda e: len(transitions(e)[0]))
    _, replay, _, _ = transitions(longest)
    payload = {'models': [m.state_dict() for m in models], 'seed': seed,
               'source': manifest['source'], 'collector': manifest['collector'],
               'calibration': manifest['calibration'],
               'manifest_sha256': sha256(Path(dataset)/'manifest.json'),
               'train_ids': [e['id'] for e in train_eps], 'validation_ids': [e['id'] for e in validation_eps],
               'test_ids': [e['id'] for e in test_eps], 'prior_relative': torch.tensor(prior_r, dtype=torch.float32),
               'prior_action': torch.tensor(prior_a, dtype=torch.float32),
               'replay_actions': torch.tensor(replay, dtype=torch.float32)}
    world = WorldModel(models, payload)
    vs, va, vy = arrays(validation_eps)
    pred, scale = world.predict(vs[:, :2]-vs[:, 2:], va)
    ratio = np.linalg.norm(pred-vy, axis=1)/scale
    # Empirical 90th percentile; temporally correlated, tiny samples: no formal guarantee.
    world.conformal_scale = float(np.quantile(ratio, 0.9, method='higher'))
    payload['conformal_scale'] = world.conformal_scale
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out/'world_model.pt')
    report = {'source': manifest['source'], 'collector': manifest['collector'], 'split': {k: payload[k] for k in ['train_ids', 'validation_ids', 'test_ids']},
              'calibration': manifest['calibration'],
              'coordinate_units': ('normalised simulator metres, not measured real metres' if manifest['calibration'].get('metric_scale_measured') is False else 'calibrated tabletop metres'),
              'training_seconds': time.perf_counter()-started, 'parameters_per_member': sum(p.numel() for p in models[0].parameters()),
              'epochs_max': epochs, 'ensemble': ensemble, 'seed': seed,
              'learned_contact_radius_mm': [float((0.008+0.037*torch.sigmoid(m.range_logit)).detach())*1000 for m in models],
              'validation': prediction_report(world, validation_eps), 'test': prediction_report(world, test_eps),
              'limitations': ['Tracked planar states, not video generation.', 'Validation radius is empirical; no coverage guarantee for correlated clips or simulation shift.',
                             'Simulation uses ground-truth state observations.', 'No sim policy training: the planner transfers the real-data dynamics model directly.']}
    write_json(out/'prediction_metrics.json', report)
    print(f'Saved {out}/world_model.pt. Held-out test error: {report["test"]["mean_error_mm"]:.2f} mm')
    return report
