from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from speech_mtl.common.data.collate import (
    pad_cached_feature_batch,
)
from speech_mtl.common.models.factory import build_exp61_model
from speech_mtl.common.training.amp import AmpController
from speech_mtl.common.training.optimization import (
    ThreePhaseLRScheduler,
    build_optimizer,
)
from speech_mtl.common.training.progress import ProgressWriter
from speech_mtl.common.training.state import TrainingState

from .batching import StableFrameBudgetBatchSampler
from .epoch import train_epoch_with_metrics
from .evaluation import evaluate_split_with_loss


class _SyntheticDataset(Dataset):
    languages = (
        "bh", "bn", "ch", "kn",
        "mg", "mr", "mt", "te",
    )
    lid_map = {
        "bn": 0,
        "bh": 1,
        "mg": 2,
        "mt": 3,
        "mr": 4,
        "kn": 5,
        "te": 6,
        "ch": 7,
    }
    dialect_counts = {
        "bh": 3,
        "bn": 5,
        "ch": 4,
        "kn": 5,
        "mg": 4,
        "mr": 4,
        "mt": 4,
        "te": 4,
    }

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        language = self.languages[index]
        frames = 20 + index
        generator = torch.Generator().manual_seed(
            2000 + index
        )

        return {
            "features": torch.randn(
                frames,
                768,
                generator=generator,
            ),
            "utt_id": f"synthetic_{index}",
            "num_frames": frames,
            "lid": self.lid_map[language],
            "dialect": index % self.dialect_counts[language],
            "gender": index % 2,
            "language": language,
            "filepath": f"/tmp/synthetic_{index}.wav",
            "audio_samples": frames * 320,
            "dataset_index": index,
        }


def validate_exp61_integration_contract() -> dict[str, Any]:
    torch.manual_seed(1337)

    dataset = _SyntheticDataset()
    lengths = [
        int(dataset[index]["num_frames"])
        for index in range(len(dataset))
    ]
    sampler = StableFrameBudgetBatchSampler(
        lengths,
        max_batch_frames=10_000,
        max_batch_size=8,
        number_of_buckets=2,
        shuffle=True,
        seed=1337,
    )

    count_by_epoch = []

    for epoch in (1, 2, 40):
        sampler.set_epoch(epoch)
        batches = list(iter(sampler))
        count_by_epoch.append(len(batches))

        flattened = sorted(
            index
            for batch in batches
            for index in batch
        )

        if flattened != list(range(len(dataset))):
            raise RuntimeError(
                "Stable sampler does not cover every example once"
            )

    if len(set(count_by_epoch)) != 1:
        raise RuntimeError(
            "Stable sampler batch count changed across epochs"
        )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=pad_cached_feature_batch,
        num_workers=0,
    )

    model = build_exp61_model()
    optimizer = build_optimizer(model)
    scheduler = ThreePhaseLRScheduler(
        optimizer,
        optimizer_steps_per_epoch=len(dataloader),
    )
    amp = AmpController(
        device="cpu",
        enabled=False,
    )
    state = TrainingState()

    with tempfile.TemporaryDirectory() as directory:
        progress = ProgressWriter(
            Path(directory) / "progress.json"
        )

        train_report = train_epoch_with_metrics(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_controller=amp,
            device=torch.device("cpu"),
            epoch=1,
            training_state=state,
            gradient_accumulation_steps=1,
            maximum_gradient_norm=1.0,
            progress_writer=progress,
            progress_interval_batches=1,
            show_progress_bar=False,
        )

        evaluation_report = evaluate_split_with_loss(
            model=model,
            dataloader=dataloader,
            device=torch.device("cpu"),
            use_amp=False,
            predictions_path=(
                Path(directory) / "predictions.csv"
            ),
            show_progress_bar=False,
        )

        if not (
            Path(directory) / "predictions.csv"
        ).is_file():
            raise RuntimeError(
                "Integration prediction CSV was not created"
            )

    if train_report["examples"] != len(dataset):
        raise RuntimeError(
            "Training report example count mismatch"
        )

    if evaluation_report["examples"] != len(dataset):
        raise RuntimeError(
            "Evaluation report example count mismatch"
        )

    expected_optimizer_steps = len(dataloader)

    if state.optimizer_step != expected_optimizer_steps:
        raise RuntimeError(
            "Unexpected optimizer-step count: "
            f"expected={expected_optimizer_steps}, "
            f"actual={state.optimizer_step}"
        )

    return {
        "status": "PASS",
        "stable_sampler_batches": count_by_epoch[0],
        "synthetic_examples": len(dataset),
        "optimizer_steps": state.optimizer_step,
        "train_selection_metric": train_report[
            "summary"
        ]["selection_metric"],
        "evaluation_selection_metric": (
            evaluation_report["summary"][
                "selection_metric"
            ]
        ),
        "training_integration": "PASS",
        "evaluation_integration": "PASS",
        "prediction_export": "PASS",
    }
