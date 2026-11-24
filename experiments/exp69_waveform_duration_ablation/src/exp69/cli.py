from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .cache import build_logmel_cache
from .config import load_config
from .evaluate import evaluate_condition
from .preflight import run_preflight
from .report import generate_report
from .train import train_condition
from .utils import atomic_json, utc_now
from .verify import verify_experiment


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "EXP 6.9 requires exactly one visible GPU; physical GPU 1 must map to local cuda:0"
        )
    return torch.device("cuda:0")


def run_queue(config) -> None:
    device = _device()
    total = len(config.domains) * len(config.durations)
    index = 0
    queue_path = config.path("run_root") / "QUEUE_PROGRESS.json"
    for domain in config.domains:
        for duration in config.durations:
            index += 1
            atomic_json(
                queue_path,
                {
                    "status": "running",
                    "index": index,
                    "total": total,
                    "domain": domain,
                    "duration": duration,
                    "stage": "training",
                    "updated_utc": utc_now(),
                },
            )
            train_condition(config, domain=domain, duration_name=duration, device=device)

            # Matched development evaluation at the training duration.
            atomic_json(
                queue_path,
                {
                    "status": "running",
                    "index": index,
                    "total": total,
                    "domain": domain,
                    "duration": duration,
                    "stage": "evaluation",
                    "updated_utc": utc_now(),
                },
            )
            evaluate_condition(
                config,
                train_domain=domain,
                train_duration=duration,
                input_domain=domain,
                split="dev",
                test_duration=duration,
                device=device,
            )
            # Same-precision duration transfer, including the matched cell.
            for test_duration in config.durations:
                evaluate_condition(
                    config,
                    train_domain=domain,
                    train_duration=duration,
                    input_domain=domain,
                    split="test",
                    test_duration=test_duration,
                    device=device,
                )
            # Direct cross-precision evaluation at the training duration only.
            other = "sixteen_bit" if domain == "one_bit" else "one_bit"
            evaluate_condition(
                config,
                train_domain=domain,
                train_duration=duration,
                input_domain=other,
                split="test",
                test_duration=duration,
                device=device,
            )
            atomic_json(
                config.path("run_root") / domain / duration / "CONDITION_COMPLETE.json",
                {
                    "status": "CONDITION_COMPLETE",
                    "created_utc": utc_now(),
                    "domain": domain,
                    "duration": duration,
                },
            )
            print(
                f"EXP69_CONDITION_COMPLETE index={index}/{total} domain={domain} duration={duration}",
                flush=True,
            )

    atomic_json(
        queue_path,
        {
            "status": "training_and_evaluation_complete",
            "completed_conditions": total,
            "total": total,
            "updated_utc": utc_now(),
        },
    )
    print("EXP69_QUEUE_TRAINING_EVALUATION_COMPLETE", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--config",
        default=(
            "/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA/"
            "configs/exp69.json"
        ),
    )
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight-cpu")
    sub.add_parser("preflight-gpu")
    sub.add_parser("build-cache")
    sub.add_parser("run-queue")
    sub.add_parser("report")
    sub.add_parser("verify")
    sub.add_parser("all")
    return result


def main() -> None:
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    if arguments.command == "preflight-cpu":
        run_preflight(config, require_gpu=False)
    elif arguments.command == "preflight-gpu":
        run_preflight(config, require_gpu=True)
    elif arguments.command == "build-cache":
        build_logmel_cache(config)
    elif arguments.command == "run-queue":
        run_queue(config)
    elif arguments.command == "report":
        generate_report(config)
    elif arguments.command == "verify":
        verify_experiment(config)
    elif arguments.command == "all":
        run_preflight(config, require_gpu=True)
        build_logmel_cache(config)
        run_queue(config)
        generate_report(config)
        verify_experiment(config)
    else:
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
