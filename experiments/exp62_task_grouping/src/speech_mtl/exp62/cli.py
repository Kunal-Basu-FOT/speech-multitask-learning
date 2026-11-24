"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import torch

from .conditions import CONDITION_ORDER, DOMAINS, get_condition
from .config import load_config
from .evaluate import evaluate_condition
from .labels import derive_label_space, load_label_space, save_label_space
from .import_exp61 import import_to_file
from .preflight import run_preflight
from .report import generate_report
from .train import train_condition
from .utils import atomic_json_dump, ensure_single_visible_cuda
from .validate import run_synthetic_validation


def _labels(cfg):
    path = cfg.paths.project_root / "validation" / "label_space.json"
    if path.is_file():
        return load_label_space(path)
    labels = derive_label_space(cfg.paths.cache_root)
    save_label_space(labels, path)
    return labels


def cmd_validate(args):
    cfg = load_config(args.config)
    labels = derive_label_space(cfg.paths.cache_root)
    save_label_space(labels, cfg.paths.project_root / "validation" / "label_space.json")
    run_synthetic_validation(cfg, torch.device("cpu"))
    if args.preflight:
        run_preflight(cfg, labels, full_hash=args.full_hash)
    marker = cfg.paths.project_root / "validation" / "EXP62_VALIDATION_PASS"
    marker.write_text("EXP62_VALIDATION_PASS\n")
    print("EXP62_VALIDATION_PASS")


def cmd_preflight(args):
    cfg = load_config(args.config)
    labels = _labels(cfg)
    result = run_preflight(cfg, labels, full_hash=args.full_hash)
    print(json.dumps(result, indent=2))
    print("EXP62_REAL_CACHE_PREFLIGHT_PASS")


def cmd_train(args):
    cfg = load_config(args.config)
    labels = _labels(cfg)
    device = ensure_single_visible_cuda() if not args.allow_cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_condition(cfg, labels, get_condition(args.condition), args.domain, device, resume=not args.no_resume)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    labels = _labels(cfg)
    device = ensure_single_visible_cuda() if not args.allow_cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluate_condition(cfg, labels, get_condition(args.condition), args.train_domain, args.test_domain, device)


def cmd_run_condition(args):
    cfg = load_config(args.config)
    labels = _labels(cfg)
    device = ensure_single_visible_cuda()
    condition = get_condition(args.condition)
    train_condition(cfg, labels, condition, args.domain, device, resume=True)
    for test_domain in DOMAINS:
        evaluate_condition(cfg, labels, condition, args.domain, test_domain, device)


def cmd_run_queue(args):
    cfg = load_config(args.config)
    marker = cfg.paths.project_root / "validation" / "EXP62_VALIDATION_PASS"
    if not marker.is_file():
        raise RuntimeError(f"Validation marker missing: {marker}")
    labels = _labels(cfg)
    run_preflight(cfg, labels, full_hash=False)
    device = ensure_single_visible_cuda()
    queue_progress = cfg.paths.run_root / "QUEUE_PROGRESS.json"
    total = len(DOMAINS) * len(CONDITION_ORDER)
    completed = 0
    for domain in DOMAINS:
        for condition_name in CONDITION_ORDER:
            condition = get_condition(condition_name)
            atomic_json_dump(
                {
                    "status": "running",
                    "domain": domain,
                    "condition": condition_name,
                    "completed_conditions": completed,
                    "total_conditions": total,
                    "timestamp": time.time(),
                },
                queue_progress,
            )
            train_condition(cfg, labels, condition, domain, device, resume=True)
            for test_domain in DOMAINS:
                complete = cfg.paths.results_root / domain / condition_name / f"test_{test_domain}" / "EVALUATION_COMPLETE.json"
                if not complete.is_file():
                    evaluate_condition(cfg, labels, condition, domain, test_domain, device)
            completed += 1
    atomic_json_dump(
        {"status": "complete", "completed_conditions": completed, "total_conditions": total, "timestamp": time.time()},
        queue_progress,
    )
    atomic_json_dump(
        {"status": "complete", "models_trained": 6, "evaluations": 12, "timestamp": time.time()},
        cfg.paths.run_root / "EXPERIMENT_COMPLETE.json",
    )
    print("EXP62_REDUCED_QUEUE_COMPLETE")


def cmd_import_exp61(args):
    cfg = load_config(args.config)
    output = Path(args.output).resolve() if args.output else (
        cfg.paths.report_root / "exp61_matched_summary.json"
    )
    imported = import_to_file(cfg.paths.exp61_results_root, output)
    print(f"Imported EXP 6.1 summary: {imported}")
    print("REVIEW_REQUIRED: verify provenance before generating comparison figures")


def cmd_report(args):
    cfg = load_config(args.config)
    output = generate_report(cfg, Path(args.exp61_summary).resolve() if args.exp61_summary else None)
    print(f"Report outputs: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp62/reduced_experiment.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--full-hash", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("preflight")
    p.add_argument("--full-hash", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("train")
    p.add_argument("--domain", choices=DOMAINS, required=True)
    p.add_argument("--condition", choices=CONDITION_ORDER, required=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="Testing only")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("--train-domain", choices=DOMAINS, required=True)
    p.add_argument("--test-domain", choices=DOMAINS, required=True)
    p.add_argument("--condition", choices=CONDITION_ORDER, required=True)
    p.add_argument("--allow-cpu", action="store_true", help="Testing only")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("run-condition")
    p.add_argument("--domain", choices=DOMAINS, required=True)
    p.add_argument("--condition", choices=CONDITION_ORDER, required=True)
    p.set_defaults(func=cmd_run_condition)

    p = sub.add_parser("run-queue")
    p.set_defaults(func=cmd_run_queue)

    p = sub.add_parser("import-exp61")
    p.add_argument("--output")
    p.set_defaults(func=cmd_import_exp61)

    p = sub.add_parser("report")
    p.add_argument("--exp61-summary")
    p.set_defaults(func=cmd_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
