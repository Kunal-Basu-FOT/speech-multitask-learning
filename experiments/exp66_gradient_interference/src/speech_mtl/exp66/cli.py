from __future__ import annotations

import argparse
import time

import torch

from .config import load_config
from .evaluate import evaluate_condition
from .labels import derive_label_space, save_label_space
from .model import CONDITIONS
from .preflight import run_preflight
from .report import generate_domain_report, generate_final_report
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
    save_label_space(labels, cfg.paths.project_root / "validation" / "label_space.json")
    return labels


def _run_condition(cfg, labels, domain: str, condition: str, device: torch.device) -> None:
    train_condition(cfg, labels, domain, condition, device)
    evaluate_condition(cfg, labels, domain, domain, condition, device)
    other = "sixteen_bit" if domain == "one_bit" else "one_bit"
    evaluate_condition(cfg, labels, domain, other, condition, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp66/experiment.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("preflight")
    domain_report = subparsers.add_parser("domain-report")
    domain_report.add_argument("--domain", choices=DOMAINS, required=True)
    subparsers.add_parser("final-report")
    condition_parser = subparsers.add_parser("run-condition")
    condition_parser.add_argument("--domain", choices=DOMAINS, required=True)
    condition_parser.add_argument("--condition", choices=CONDITIONS, required=True)
    subparsers.add_parser("run-queue")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.command == "validate":
        run_validation(args.config)
        return
    if args.command == "preflight":
        run_preflight(cfg, _labels(cfg))
        print("EXP66_REAL_CACHE_PREFLIGHT_PASS")
        return
    if args.command == "domain-report":
        generate_domain_report(cfg, args.domain)
        return
    if args.command == "final-report":
        generate_final_report(cfg)
        return

    labels = _labels(cfg)
    device = _device()
    if args.command == "run-condition":
        _run_condition(cfg, labels, args.domain, args.condition, device)
        return

    queue_progress = cfg.paths.run_root / "QUEUE_PROGRESS.json"
    run_preflight(cfg, labels)
    total_conditions = len(DOMAINS) * len(CONDITIONS)
    global_index = 0

    # Deliberate queue semantics requested by the user:
    # finish every 1-bit method, save every evaluation/table/figure, and only
    # then permit the first 16-bit training run to begin.
    for domain_index, domain in enumerate(DOMAINS, start=1):
        for condition_index, condition in enumerate(CONDITIONS, start=1):
            global_index += 1
            atomic_json_dump(
                {
                    "status": "running_condition",
                    "global_index": global_index,
                    "global_total": total_conditions,
                    "domain_index": domain_index,
                    "domain_total": len(DOMAINS),
                    "condition_index_within_domain": condition_index,
                    "condition_total_within_domain": len(CONDITIONS),
                    "domain": domain,
                    "condition": condition,
                    "includes_matched_evaluation": True,
                    "includes_cross_precision_evaluation": True,
                    "next_domain_is_blocked_until_domain_report_complete": True,
                    "timestamp": time.time(),
                },
                queue_progress,
            )
            _run_condition(cfg, labels, domain, condition, device)

        atomic_json_dump(
            {
                "status": "generating_domain_report_before_next_domain",
                "domain": domain,
                "completed_conditions": list(CONDITIONS),
                "timestamp": time.time(),
            },
            queue_progress,
        )
        report_root = generate_domain_report(cfg, domain)
        atomic_json_dump(
            {
                "status": "complete",
                "domain": domain,
                "conditions": list(CONDITIONS),
                "matched_and_cross_precision_complete": True,
                "report_root": str(report_root),
                "timestamp": time.time(),
            },
            cfg.paths.run_root / domain / "DOMAIN_COMPLETE.json",
        )
        atomic_json_dump(
            {
                "status": "domain_saved",
                "domain": domain,
                "domain_report_complete": True,
                "next_domain_may_start": domain != DOMAINS[-1],
                "timestamp": time.time(),
            },
            queue_progress,
        )
        print(f"EXP66_DOMAIN_FULLY_SAVED domain={domain}", flush=True)

    generate_final_report(cfg)
    atomic_json_dump(
        {
            "status": "complete",
            "queue_order": [
                "one_bit/standard_logged",
                "one_bit/pcgrad",
                "one_bit/gradnorm",
                "SAVE_ONE_BIT_RESULTS_TABLES_FIGURES",
                "sixteen_bit/standard_logged",
                "sixteen_bit/pcgrad",
                "sixteen_bit/gradnorm",
                "SAVE_SIXTEEN_BIT_RESULTS_TABLES_FIGURES",
                "GENERATE_FINAL_COMPARISON",
            ],
            "domains": list(DOMAINS),
            "conditions": list(CONDITIONS),
            "timestamp": time.time(),
        },
        queue_progress,
    )
    atomic_json_dump(
        {"status": "complete", "timestamp": time.time()},
        cfg.paths.run_root / "EXPERIMENT_COMPLETE.json",
    )
    print("EXP66_QUEUE_COMPLETE")


if __name__ == "__main__":
    main()
