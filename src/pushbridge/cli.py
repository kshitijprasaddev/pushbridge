import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description='PushBridge: personal pushing videos to learned dynamics and a simulated robot.')
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('serve', help='Open the static research dashboard with a local Python server')
    c.add_argument('--port', type=int, default=8767)
    c.add_argument('--directory', type=Path, default=Path('.'))
    c = commands.add_parser('calibrate', help='Click the four A4 sheet corners in a browser')
    c.add_argument('--video', required=True, type=Path)
    c.add_argument('--out', default=Path('data/calibration.json'), type=Path)
    c.add_argument('--port', type=int, default=8765)
    c.add_argument('--corners', help='Alternative: "x,y x,y x,y x,y", in TL TR BR BL order')
    c = commands.add_parser('ingest', help='Track red object and blue finger; inspect the preview videos')
    c.add_argument('--videos', required=True, type=Path)
    c.add_argument('--calibration', required=True, type=Path)
    c.add_argument('--out', default=Path('data/processed/personal'), type=Path)
    c.add_argument('--collector', required=True, help='Who recorded these videos')
    c.add_argument('--source', choices=['personal', 'synthetic'], required=True)
    c = commands.add_parser('train', help='Fit contact dynamics; split by whole clips')
    c.add_argument('--dataset', required=True, type=Path)
    c.add_argument('--out', default=Path('results/personal/model'), type=Path)
    c.add_argument('--epochs', type=int, default=200)
    c.add_argument('--seed', type=int, default=7)
    c.add_argument('--ensemble', type=int, default=3)
    c = commands.add_parser('evaluate', help='Run paired simulation trials and generate a report')
    c.add_argument('--checkpoint', required=True, type=Path)
    c.add_argument('--out', default=Path('results/personal/evaluation'), type=Path)
    c.add_argument('--episodes', type=int, default=20)
    c.add_argument('--steps', type=int, default=120)
    c.add_argument('--seed', type=int, default=2026)
    c.add_argument('--no-render', action='store_true')
    c.add_argument('--modes', nargs='+', choices=['replay', 'prior', 'geometry', 'mpc', 'mpc_no_uncertainty'])
    c = commands.add_parser('train-rl', help='Train state-based SAC, optionally initialized from training-only personal actions')
    c.add_argument('--dataset', required=True, type=Path)
    c.add_argument('--checkpoint', required=True, type=Path, help='World model fixing the personal-data split')
    c.add_argument('--out', required=True, type=Path)
    c.add_argument('--steps', type=int, default=40000)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--demo-init', action='store_true')
    c.add_argument('--eval-interval', type=int, default=5000)
    c = commands.add_parser('transfer-experiment', help='Post-train source policies on a compact three-joint embodiment and evaluate retargeting')
    c.add_argument('--source-root', type=Path, default=Path('results/rl_comparison'))
    c.add_argument('--checkpoint', type=Path, default=Path('results/personal_five/model/world_model.pt'))
    c.add_argument('--out', type=Path, default=Path('results/embodiment_transfer'))
    c.add_argument('--steps', type=int, default=20000)
    c.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    c.add_argument('--episodes', type=int, default=20)
    c = commands.add_parser('rl-experiment', help='Compare scratch and personal-demo initialized SAC across seeds against MPC/geometry')
    c.add_argument('--dataset', required=True, type=Path)
    c.add_argument('--checkpoint', required=True, type=Path)
    c.add_argument('--out', required=True, type=Path)
    c.add_argument('--steps', type=int, default=40000)
    c.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    c.add_argument('--eval-interval', type=int, default=5000)
    c = commands.add_parser('synthetic', help='Generate SYNTHETIC fixtures only; does not satisfy the challenge')
    c.add_argument('--out', default=Path('data/raw/synthetic'), type=Path)
    c.add_argument('--episodes', type=int, default=10)
    c.add_argument('--seed', type=int, default=19)
    c = commands.add_parser('retarget-demo', help='Drive the arm from one personal clip using its drawn black target circle')
    c.add_argument('--video', required=True, type=Path)
    c.add_argument('--out', default=Path('results/personal_retarget'), type=Path)
    c.add_argument('--max-steps', type=int, default=600)
    c = commands.add_parser('ingest-circle', help='Prepare personal training clips using the drawn black circle instead of paper corners')
    c.add_argument('--videos', required=True, type=Path, help='One video or a folder of independent clips')
    c.add_argument('--out', default=Path('data/processed/personal_circle'), type=Path)
    c.add_argument('--collector', required=True)
    c = commands.add_parser('smoke', help='Synthetic end-to-end software check, with prominent provenance labels')
    c.add_argument('--out', default=Path('results/smoke'), type=Path)
    c.add_argument('--episodes', type=int, default=8, help='Number of paired evaluation trials')
    c.add_argument('--epochs', type=int, default=80)
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == 'serve':
            import http.server
            from functools import partial
            handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(args.directory.resolve()))
            with http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
                print(f'PushBridge dashboard: http://127.0.0.1:{args.port}/', flush=True)
                server.serve_forever()
        elif args.command == 'calibrate':
            from .calibration import calibrate
            corners = [[float(v) for v in point.split(',')] for point in args.corners.split()] if args.corners else None
            calibrate(args.video, args.out, corners, args.port)
        elif args.command == 'ingest':
            from .tracking import ingest
            ingest(args.videos, args.calibration, args.out, args.collector, args.source)
        elif args.command == 'train':
            from .model import train
            train(args.dataset, args.out, args.epochs, args.seed, args.ensemble)
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            evaluate(args.checkpoint, args.out, args.episodes, args.steps, args.seed, not args.no_render, args.modes)
        elif args.command == 'train-rl':
            from .rl import train_sac
            train_sac(args.dataset, args.checkpoint, args.out, args.steps, args.seed, args.demo_init, args.eval_interval)
        elif args.command == 'rl-experiment':
            from .rl import experiment
            experiment(args.dataset, args.checkpoint, args.out, args.steps, args.seeds, args.eval_interval)
        elif args.command == 'transfer-experiment':
            from .transfer import transfer_experiment
            transfer_experiment(args.source_root, args.checkpoint, args.out, args.steps, args.seeds, args.episodes)
        elif args.command == 'synthetic':
            from .synthetic import make_synthetic
            make_synthetic(args.out, args.episodes, args.seed)
        elif args.command == 'retarget-demo':
            from .retarget import retarget_demo
            retarget_demo(args.video, args.out, args.max_steps)
        elif args.command == 'ingest-circle':
            from .retarget import ingest_circle
            ingest_circle(args.videos, args.out, args.collector)
        elif args.command == 'smoke':
            from .synthetic import make_synthetic
            from .tracking import ingest
            from .model import train
            from .evaluation import evaluate
            raw, processed, fitted, evaluation = [args.out/n for n in ['synthetic_raw', 'synthetic_processed', 'model', 'evaluation']]
            make_synthetic(raw)
            ingest(raw, raw/'calibration.json', processed, 'synthetic generator, not applicant', 'synthetic')
            train(processed, fitted, args.epochs)
            evaluate(fitted/'world_model.pt', evaluation, args.episodes)
            print(f'SYNTHETIC CHECK ONLY. Open {evaluation}/report.html')
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f'Error: {e}') from e


if __name__ == '__main__':
    main()
