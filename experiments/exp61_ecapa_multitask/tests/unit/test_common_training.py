from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from speech_mtl.common.training.amp import AmpController
from speech_mtl.common.training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from speech_mtl.common.training.losses import (
    compute_oracle_dialect_loss,
)
from speech_mtl.common.training.optimization import (
    SchedulerConfig,
    ThreePhaseLRScheduler,
)
from speech_mtl.common.training.phases import (
    PhaseController,
)
from speech_mtl.common.training.state import TrainingState


class MinimalPhaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.head = nn.Linear(3, 2)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)


class CommonTrainingTests(unittest.TestCase):
    def test_phase_boundaries(self) -> None:
        controller = PhaseController()

        self.assertEqual(
            controller.for_epoch(1).name,
            "phase1_warmup_joint",
        )
        self.assertEqual(
            controller.for_epoch(5).name,
            "phase1_warmup_joint",
        )
        self.assertEqual(
            controller.for_epoch(6).name,
            "phase2_full_joint",
        )
        self.assertEqual(
            controller.for_epoch(35).name,
            "phase2_full_joint",
        )
        self.assertEqual(
            controller.for_epoch(36).name,
            "phase3_heads_only",
        )
        self.assertEqual(
            controller.for_epoch(40).name,
            "phase3_heads_only",
        )

        with self.assertRaises(ValueError):
            controller.for_epoch(0)

        with self.assertRaises(ValueError):
            controller.for_epoch(41)

    def test_phase3_freezes_only_encoder(self) -> None:
        model = MinimalPhaseModel()
        controller = PhaseController()
        model.train()
        controller.apply(model, 36)

        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.head.parameters()
            )
        )
        self.assertFalse(model.encoder.training)

    def test_scheduler_contract_and_roundtrip(self) -> None:
        encoder_parameter = nn.Parameter(
            torch.tensor(1.0)
        )
        task_parameter = nn.Parameter(
            torch.tensor(2.0)
        )

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [encoder_parameter],
                    "group_name": "encoder",
                },
                {
                    "params": [task_parameter],
                    "group_name": "tasks",
                },
            ],
            lr=1.0e-4,
        )

        scheduler = ThreePhaseLRScheduler(
            optimizer,
            optimizer_steps_per_epoch=2,
            config=SchedulerConfig(),
        )

        self.assertEqual(
            scheduler.current_learning_rates(),
            {"encoder": 1.0e-5, "tasks": 1.0e-5},
        )

        for _ in range(10):
            scheduler.step()

        saved = scheduler.state_dict()

        restored = ThreePhaseLRScheduler(
            optimizer,
            optimizer_steps_per_epoch=2,
            config=SchedulerConfig(),
        )
        restored.load_state_dict(saved)

        self.assertEqual(
            restored.state_dict(),
            saved,
        )

        phase3 = restored.learning_rates_for_step(70)
        self.assertEqual(
            phase3,
            {"encoder": 0.0, "tasks": 2.0e-5},
        )

    def test_oracle_dialect_loss_covers_batch(self) -> None:
        logits = {
            "bh": torch.tensor(
                [[2.0, 0.0, -1.0]],
                requires_grad=True,
            ),
            "bn": torch.tensor(
                [[0.0, 2.0, -1.0, -2.0, -3.0]],
                requires_grad=True,
            ),
        }
        indices = {
            "bh": torch.tensor([0], dtype=torch.long),
            "bn": torch.tensor([1], dtype=torch.long),
        }
        targets = torch.tensor(
            [0, 1],
            dtype=torch.long,
        )

        loss, by_language, examples = (
            compute_oracle_dialect_loss(
                dialect_logits=logits,
                dialect_batch_indices=indices,
                dialect_targets=targets,
            )
        )

        self.assertEqual(examples, 2)
        self.assertEqual(set(by_language), {"bh", "bn"})
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertIsNotNone(logits["bh"].grad)
        self.assertIsNotNone(logits["bn"].grad)

    def test_checkpoint_roundtrip(self) -> None:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(model.parameters()),
                    "group_name": "encoder",
                },
                {
                    "params": [],
                    "group_name": "tasks",
                },
            ],
            lr=1.0e-4,
        )
        scheduler = ThreePhaseLRScheduler(
            optimizer,
            optimizer_steps_per_epoch=1,
        )
        amp = AmpController(
            device="cpu",
            enabled=False,
        )
        state = TrainingState(
            epoch=3,
            global_batch=7,
            optimizer_step=4,
            phase_name="phase1_warmup_joint",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"

            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                amp_controller=amp,
                state=state,
                extra={"key": "value"},
            )

            restored_model = nn.Linear(4, 2)
            restored_optimizer = torch.optim.AdamW(
                [
                    {
                        "params": list(
                            restored_model.parameters()
                        ),
                        "group_name": "encoder",
                    },
                    {
                        "params": [],
                        "group_name": "tasks",
                    },
                ],
                lr=1.0e-4,
            )
            restored_scheduler = ThreePhaseLRScheduler(
                restored_optimizer,
                optimizer_steps_per_epoch=1,
            )
            restored_amp = AmpController(
                device="cpu",
                enabled=False,
            )

            restored_state, extra = (
                load_training_checkpoint(
                    path,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    scheduler=restored_scheduler,
                    amp_controller=restored_amp,
                    restore_rng=False,
                )
            )

            self.assertEqual(
                restored_state.to_dict(),
                state.to_dict(),
            )
            self.assertEqual(extra, {"key": "value"})

            for original, restored in zip(
                model.parameters(),
                restored_model.parameters(),
            ):
                self.assertTrue(
                    torch.equal(original, restored)
                )


if __name__ == "__main__":
    unittest.main()
