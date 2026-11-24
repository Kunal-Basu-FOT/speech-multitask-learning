from __future__ import annotations

import argparse
import time

import torch

from .config import load_config
from .evaluate import evaluate_condition
from .labels import derive_label_space, save_label_space
from .model import CONDITION_NAME
from .preflight import run_preflight
from .report import generate_report
from .train import train_condition
from .utils import atomic_json_dump
from .validate import run as run_validation

DOMAINS = ("one_bit", "sixteen_bit")


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Exactly one CUDA device must be visible, got {torch.cuda.device_count()}"
        )
    return torch.device("cuda:0")


def _labels(cfg):
    labels = derive_label_space(cfg.paths.cache_root)
    save_label_space(
        labels,
        cfg.paths.project_root / "validation" / "label_space.json",
    )
    return labels


def _run_domain(cfg, labels, domain: str, device: torch.device) -> None:
    train_condition(cfg, labels, domain, device)
    evaluate_condition(cfg, labels, domain, domain, device)
    other_domain = "sixteen_bit" if domain == "one_bit" else "one_bit"
    evaluate_condition(cfg, labels, domain, other_domain, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/exp67/experiment.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("preflight")
    subparsers.add_parser("report")
    condition_parser = subparsers.add_parser("run-condition")
    condition_parser.add_argument("--domain", choices=DOMAINS, required=True)
    subparsers.add_parser("run-queue")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.command == "validate":
        run_validation(args.config)
        return
    if args.command == "preflight":
        run_preflight(cfg, _labels(cfg))
        print("EXP67_REAL_CACHE_PREFLIGHT_PASS")
        return
    if args.command == "report":
        generate_report(cfg)
        return

    labels = _labels(cfg)
    device = _device()
    if args.command == "run-condition":
        _run_domain(cfg, labels, args.domain, device)
        return

    queue_progress = cfg.paths.run_root / "QUEUE_PROGRESS.json"
    run_preflight(cfg, labels)
    for index, domain in enumerate(DOMAINS, start=1):
        atomic_json_dump(
            {
                "status": "running",
                "index": index,
                "total": len(DOMAINS),
                "condition": CONDITION_NAME,
                "domain": domain,
                "includes_matched_evaluation": True,
                "includes_cross_precision_evaluation": True,
                "timestamp": time.time(),
            },
            queue_progress,
        )
        _run_domain(cfg, labels, domain, device)

    generate_report(cfg)
    atomic_json_dump(
        {
            "status": "complete",
            "condition": CONDITION_NAME,
            "domains": list(DOMAINS),
            "evaluation_cells": [
                "one_bit->one_bit",
                "one_bit->sixteen_bit",
                "sixteen_bit->sixteen_bit",
                "sixteen_bit->one_bit",
            ],
            "timestamp": time.time(),
        },
        queue_progress,
    )
    atomic_json_dump(
        {
            "status": "complete",
            "timestamp": time.time(),
        },
        cfg.paths.run_root / "EXPERIMENT_COMPLETE.json",
    )
    print("EXP67_QUEUE_COMPLETE")


if __name__ == "__main__":
    main()
