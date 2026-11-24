from __future__ import annotations

import unittest

import torch

from speech_mtl.common.models.audit import audit_model_parameters
from speech_mtl.common.models.factory import build_exp61_model
from speech_mtl.common.models.uncertainty import (
    HomoscedasticTaskWeighting,
)


class CommonModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(1337)
        cls.model = build_exp61_model()

    def test_exact_parameter_contract(self) -> None:
        report = audit_model_parameters(
            self.model,
            enforce_exact=True,
        )

        self.assertEqual(report.ecapa, 7_212_660)
        self.assertEqual(report.total, 7_723_170)
        self.assertEqual(report.task_weighting, 3)

    def test_forward_and_oracle_routing(self) -> None:
        model = self.model
        model.eval()

        languages = ["bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"]
        frame_lengths = torch.tensor(
            [48, 47, 46, 45, 44, 43, 42, 41],
            dtype=torch.long,
        )
        maximum_frames = 48
        relative_lengths = (
            frame_lengths.to(torch.float32) / maximum_frames
        )
        valid_mask = (
            torch.arange(maximum_frames).unsqueeze(0)
            < frame_lengths.unsqueeze(1)
        )

        features = torch.randn(8, maximum_frames, 768)
        features = features.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0,
        )

        with torch.no_grad():
            output = model(
                features=features,
                relative_lengths=relative_lengths,
                oracle_languages=languages,
                valid_mask=valid_mask,
            )

        self.assertEqual(
            tuple(output.embeddings.shape),
            (8, 192),
        )
        self.assertEqual(
            tuple(output.language_logits.shape),
            (8, 8),
        )
        self.assertEqual(
            tuple(output.gender_logits.shape),
            (8, 2),
        )

        expected = {
            "bh": 3,
            "bn": 5,
            "ch": 4,
            "kn": 5,
            "mg": 4,
            "mr": 4,
            "mt": 4,
            "te": 4,
        }

        for language, classes in expected.items():
            self.assertEqual(
                tuple(output.dialect_logits[language].shape),
                (1, classes),
            )

    def test_uncertainty_weighting_has_gradients(self) -> None:
        weighting = HomoscedasticTaskWeighting()
        losses = {
            "language": torch.tensor(1.0, requires_grad=True),
            "dialect": torch.tensor(2.0, requires_grad=True),
            "gender": torch.tensor(3.0, requires_grad=True),
        }

        total, terms = weighting(losses)
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertEqual(
            set(terms),
            {"language", "dialect", "gender"},
        )

        for parameter in weighting.parameters():
            self.assertIsNotNone(parameter.grad)

    def test_encoder_freeze_and_unfreeze(self) -> None:
        model = build_exp61_model()

        model.set_encoder_trainable(False)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.task_parameters()
            )
        )

        model.set_encoder_trainable(True)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.encoder.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
