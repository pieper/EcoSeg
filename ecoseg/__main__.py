"""CLI entry point for EcoSeg.

Usage:
    python -m ecoseg serve [--port 8080] [--data-root /path/to/data]
    python -m ecoseg run --data-root /path/to/data [--config experiment.json]
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(prog="ecoseg", description="EcoSeg Server")
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the EcoSeg server")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--data-root", type=str, default=None)
    serve_parser.add_argument("--ohif-dir", type=str, default=None)

    # run
    run_parser = subparsers.add_parser("run", help="Run an experiment directly")
    run_parser.add_argument("--data-root", type=str, required=True)
    run_parser.add_argument("--config", type=str, default=None)
    run_parser.add_argument("--output-dir", type=str, default="output")
    run_parser.add_argument("--device", type=str, default="auto")
    run_parser.add_argument("--architecture", type=str, default="cnn3",
                            choices=["cnn3", "resnet"],
                            help="Species model architecture")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        from ecoseg.server.app import create_app
        create_app(data_root=args.data_root, ohif_dir=args.ohif_dir)
        uvicorn.run(
            "ecoseg.server.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )

    elif args.command == "run":
        from ecoseg.experiments.runner import ExperimentRunner, ExperimentConfig

        if args.config:
            config = ExperimentConfig.from_json(Path(args.config))
        else:
            config = ExperimentConfig()

        config.data_root = args.data_root
        config.output_dir = args.output_dir
        config.device = args.device
        config.architecture = args.architecture

        runner = ExperimentRunner(config)
        results = runner.run_full_experiment()

        print(f"\nExperiment complete: {len(results)} generations")
        for r in results:
            print(
                f"  Gen {r.generation}: "
                f"Dice={r.mean_dice:.4f}, "
                f"ASSD={r.mean_assd:.2f}mm "
                f"({r.num_training_scans} training scans)"
            )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
