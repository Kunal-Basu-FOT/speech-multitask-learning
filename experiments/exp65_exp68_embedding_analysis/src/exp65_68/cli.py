from __future__ import annotations

import argparse

import torch

from . import __version__
from .analysis import run as run_analysis
from .config import AnalysisConfig
from .extract import run as run_extraction
from .extract import smoke as run_smoke
from .preflight import run as run_preflight
from .verify import run as run_verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP 6.5 + EXP 6.8 frozen analysis")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--skip-checkpoint-load", action="store_true")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--device", default="cuda:0")
    extract = subparsers.add_parser("extract")
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--workers", type=int, default=None)
    subparsers.add_parser("analyze")
    subparsers.add_parser("verify")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = AnalysisConfig.load()
    if arguments.command == "preflight":
        run_preflight(config, validate_checkpoints=not arguments.skip_checkpoint_load)
    elif arguments.command == "smoke":
        run_smoke(config, torch.device(arguments.device))
    elif arguments.command == "extract":
        workers = config.workers if arguments.workers is None else arguments.workers
        run_extraction(config, torch.device(arguments.device), workers)
    elif arguments.command == "analyze":
        run_analysis(config)
    elif arguments.command == "verify":
        run_verification(config)
    else:
        raise RuntimeError(f"unknown command: {arguments.command}")


if __name__ == "__main__":
    main()
