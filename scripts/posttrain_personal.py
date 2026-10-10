"""Fixed supervised personal-data post-training after simulator SAC training."""
import argparse
from pathlib import Path
import time

import torch
from stable_baselines3 import SAC

from pushbridge.data import sha256, write_json
from pushbridge.evaluation import evaluate
from pushbridge.rl import demonstration_arrays, initialize_actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('data/processed/personal_five'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/personal_five/model/world_model.pt'))
    parser.add_argument('--source-root', type=Path, default=Path('results/rl_comparison'))
    parser.add_argument('--out', type=Path, default=Path('results/personal_posttraining'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'protocol.json', {'source': 'Scratch SAC after 40,000 simulator decisions per seed',
                'seeds': [0, 1, 2], 'supervised_updates': 200, 'learning_rate': .0001, 'batch_size': 128,
                'personal_data': 'Same three training clips; validation/test clips excluded',
                'objective': 'Actor mean action MSE to observed, norm-clipped, rotation-augmented fingertip actions',
                'selection': 'Fixed update budget, no validation or test model selection',
                'evaluation_seed': 2026, 'evaluation_episodes': 20,
                'new_sim_training_interactions': 0, 'critic': 'Unchanged, not used for supervised fitting',
                'limitations': ['Human action labels can conflict with robot physics',
                                'Different personal-data timing, not a matched end-to-end algorithm comparison',
                                'No visual encoder or pretrained VLA; source pretraining is simulator SAC']})
    torch.set_num_threads(2)
    models = {}
    for seed in [0, 1, 2]:
        out = args.out/f'posttrained_seed{seed}'
        if (out/'sac.zip').exists():
            raise ValueError('Post-trained checkpoint exists; choose a new output folder')
        out.mkdir(parents=True, exist_ok=True)
        source = args.source_root/f'sac_scratch_seed{seed}/sac.zip'
        agent = SAC.load(source, device='cpu')
        obs, actions, info = demonstration_arrays(args.dataset, args.checkpoint, seed)
        before = {k: v.clone() for k,v in agent.critic.state_dict().items()}
        started = time.perf_counter()
        supervised = initialize_actor(agent, obs, actions, updates=200, seed=seed, learning_rate=.0001)
        if any(not torch.equal(before[k], v) for k,v in agent.critic.state_dict().items()):
            raise RuntimeError('Supervised actor post-training changed the critic')
        agent.save(out/'sac')
        write_json(out/'training_metrics.json', {'seed': seed, 'source_checkpoint_sha256': sha256(source),
                   'seconds': time.perf_counter()-started, 'personal_data': info, 'supervised': supervised,
                   'critic_unchanged': True, 'extra_sim_training_steps': 0})
        models[f'personal_posttrained_seed{seed}'] = out/'sac.zip'
    evaluate(args.checkpoint, args.out/'evaluation', episodes=20, seed=2026, modes=list(models), rl_models=models)


if __name__ == '__main__':
    main()
