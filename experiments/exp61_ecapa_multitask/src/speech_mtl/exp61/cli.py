from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch

from speech_mtl.common.utils.atomic_io import atomic_write_json
from speech_mtl.common.utils.device import (
    assess_physical_gpu,
    verify_single_visible_cuda_device,
)

from .config import load_exp61_config
from .preflight import run_exp61_preflight
from .runner import (
    evaluate_condition,
    generate_report,
    run_full_queue,
    train_domain,
)


def _require_wtc9() -> None:
    hostname = socket.gethostname()

    if hostname != "wtc9-server":
        raise RuntimeError(
            f"EXP 6.1 production commands require wtc9-server; "
            f"current host={hostname}"
        )


def _production_device(config) -> torch.device:
    _require_wtc9()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if visible != str(config.physical_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be exactly "
            f"{config.physical_gpu}; received {visible!r}"
        )

    verification = verify_single_visible_cuda_device()
    print(
        json.dumps(
            {
                "physical_gpu": config.physical_gpu,
                "process_device": verification,
            },
            indent=2,
        )
    )
    return torch.device("cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SPEECH_MTL EXP 6.1 runner"
    )
    parser.add_argument(
        "--config",
        default=(
            "/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK/"
            "configs/exp61/experiment.yaml"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser("preflight")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--domain",
        choices=("one_bit", "sixteen_bit"),
        required=True,
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--train-domain",
        choices=("one_bit", "sixteen_bit"),
        required=True,
    )
    evaluate_parser.add_argument(
        "--eval-domain",
        choices=("one_bit", "sixteen_bit"),
        required=True,
    )
    evaluate_parser.add_argument(
        "--split",
        choices=("dev", "test"),
        required=True,
    )

    subparsers.add_parser("report")
    subparsers.add_parser("queue")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = load_exp61_config(arguments.config)

    if arguments.command == "preflight":
        report = run_exp61_preflight(config)
        output = config.run_root / "PREFLIGHT.json"
        atomic_write_json(output, report)
        print(json.dumps(report, indent=2))
        print("EXP61_PREFLIGHT_PASS")
        return

    if arguments.command == "report":
        report = generate_report(config)
        print(json.dumps(report, indent=2))
        print("EXP61_REPORT_PASS")
        return

    if arguments.command == "queue":
        _require_wtc9()

        assessment = assess_physical_gpu(
            config.physical_gpu,
            maximum_used_memory_mib=1024,
            maximum_utilization_percent=10,
            reject_compute_processes=True,
        )

        if not assessment["available"]:
            raise RuntimeError(
                "Physical GPU 1 is not available: "
                + "; ".join(assessment["reasons"])
            )

        device = _production_device(config)
        report = run_full_queue(
            config,
            device=device,
        )
        print(json.dumps(report, indent=2))
        print("EXP61_QUEUE_COMPLETE")
        return

    device = _production_device(config)

    if arguments.command == "train":
        report = train_domain(
            config,
            domain=arguments.domain,
            device=device,
        )
    elif arguments.command == "evaluate":
        report = evaluate_condition(
            config,
            train_domain=arguments.train_domain,
            eval_domain=arguments.eval_domain,
            split=arguments.split,
            device=device,
        )
    else:
        raise AssertionError(arguments.command)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
