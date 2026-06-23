import unittest
import sys
import tempfile
import types

import numpy as np
import torch

silero_vad = types.ModuleType('silero_vad')
silero_vad.load_silero_vad = lambda *args, **kwargs: None
silero_vad.read_audio = lambda *args, **kwargs: None
silero_vad.get_speech_timestamps = lambda *args, **kwargs: []
sys.modules.setdefault('silero_vad', silero_vad)

from wespeaker.losses.aorc_losses import CrossAgeAggregationLoss
from wespeaker.losses.aorc_losses import OrdinalAgeLoss
from wespeaker.losses.aorc_losses import OrdinalPrototypeLoss
from wespeaker.losses.aorc_losses import SpeakerConditionedDirectionLoss
from wespeaker.bin.train import _load_age_labels
from wespeaker.dataset import processor
from wespeaker.models.aorc_modules import AORCWrapper
from wespeaker.models.aorc_modules import AgeResidualCompensation
from wespeaker.models.aorc_modules import OrdinalAgeHead
from wespeaker.models.aorc_modules import _scale_tensor_gradient
from wespeaker.models.aorc_modules import get_aorc_config


class DummyEncoder(torch.nn.Module):

    def __init__(self, input_dim=10, embed_dim=16):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        emb = self.linear(x)
        return torch.tensor(0.0, device=x.device), emb


class AORCTest(unittest.TestCase):

    def test_ordinal_age_head_outputs_distribution(self):
        head = OrdinalAgeHead(input_dim=16, num_age_groups=7)
        out = head(torch.randn(8, 16))
        self.assertEqual(out['age_embedding'].shape, (8, 16))
        self.assertEqual(out['age_distribution'].shape, (8, 7))
        self.assertFalse(torch.isnan(out['age_distribution']).any())
        self.assertTrue(
            torch.allclose(out['age_distribution'].sum(dim=1),
                           torch.ones(8),
                           atol=1e-5))

    def test_losses_are_finite(self):
        batch, dim, groups = 8, 16, 7
        z_age = torch.randn(batch, dim)
        prototypes = torch.randn(groups, dim)
        age_group = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1])
        speakers = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5])
        rank_logits = torch.randn(batch, groups - 1)

        losses = [
            OrdinalAgeLoss(groups)(rank_logits, age_group),
            OrdinalPrototypeLoss(groups)(z_age, prototypes, age_group),
            SpeakerConditionedDirectionLoss(groups)(z_age, prototypes,
                                                    speakers, age_group),
            CrossAgeAggregationLoss(groups)(torch.randn(batch, dim), speakers,
                                            age_group),
        ]
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))

    def test_pair_losses_return_zero_without_pairs(self):
        groups = 7
        speakers = torch.arange(4)
        age_group = torch.tensor([0, 1, 2, 3])
        z = torch.randn(4, 16)
        prototypes = torch.randn(groups, 16)
        direction = SpeakerConditionedDirectionLoss(groups)(z, prototypes,
                                                           speakers,
                                                           age_group)
        caa = CrossAgeAggregationLoss(groups)(z, speakers, age_group)
        self.assertEqual(direction.item(), 0.0)
        self.assertEqual(caa.item(), 0.0)

    def test_numeric_ignore_age_value_from_npy(self):
        with tempfile.NamedTemporaryFile(suffix='.npy') as f:
            np.save(f.name, {
                'utt_missing_int': np.int64(-1),
                'utt_missing_float': -1.0,
                'utt_missing_nan': np.nan,
                'utt_missing_nan_string': 'nan',
                'utt_age': 25.0,
            })
            labels = _load_age_labels(f.name, 'value', [25, 30], 3, -1)

        self.assertEqual(labels['utt_missing_int'], -1)
        self.assertEqual(labels['utt_missing_float'], -1)
        self.assertEqual(labels['utt_missing_nan'], -1)
        self.assertEqual(labels['utt_missing_nan_string'], -1)
        self.assertEqual(labels['utt_age'], 1)

    def test_speed_perturb_preserves_original_speaker_label(self):
        sample = {
            'key': 'utt1',
            'spk': 'spk1',
            'sample_rate': 16000,
            'wav': torch.randn(1, 160),
        }
        labeled = next(processor.spk_to_id(iter([sample]), {'spk1': 7}))

        old_randint = processor.random.randint
        old_apply = processor.torchaudio.sox_effects.apply_effects_tensor
        try:
            processor.random.randint = lambda *args, **kwargs: 1
            processor.torchaudio.sox_effects.apply_effects_tensor = (
                lambda wav, sample_rate, effects: (wav, sample_rate))
            augmented = next(processor.speed_perturb(iter([labeled]), 100))
        finally:
            processor.random.randint = old_randint
            processor.torchaudio.sox_effects.apply_effects_tensor = old_apply

        self.assertEqual(augmented['label'], 107)
        self.assertEqual(augmented['orig_label'], 7)

    def test_direction_scale_preserves_prototype_gradient_scale(self):
        torch.manual_seed(0)
        groups = 3
        loss_fn = SpeakerConditionedDirectionLoss(groups)
        speakers = torch.tensor([0, 0, 1, 1])
        age_group = torch.tensor([0, 2, 0, 2])

        z = torch.randn(4, 5, requires_grad=True)
        prototypes = torch.randn(groups, 5, requires_grad=True)
        loss = loss_fn(z, prototypes, speakers, age_group)
        loss.backward()
        z_grad = z.grad.detach().clone()
        proto_grad = prototypes.grad.detach().clone()

        z_scaled = z.detach().clone().requires_grad_(True)
        proto_scaled = prototypes.detach().clone().requires_grad_(True)
        scaled_loss = loss_fn(_scale_tensor_gradient(z_scaled, 4.0),
                              proto_scaled, speakers, age_group)
        scaled_loss.backward()

        self.assertTrue(torch.allclose(scaled_loss.detach(), loss.detach()))
        self.assertTrue(torch.allclose(z_scaled.grad, z_grad * 4.0, atol=1e-5))
        self.assertTrue(
            torch.allclose(proto_scaled.grad, proto_grad, atol=1e-5))

    def test_orc_shapes_and_smoothness(self):
        orc = AgeResidualCompensation(num_age_groups=7, embed_dim=16)
        q_age = torch.softmax(torch.randn(8, 7), dim=-1)
        z_spk, residual = orc(torch.randn(8, 16), q_age)
        self.assertEqual(z_spk.shape, (8, 16))
        self.assertEqual(residual.shape, (8, 16))
        self.assertTrue(torch.isfinite(orc.smoothness_loss()))

    def test_wrapper_baseline_and_inference(self):
        disabled = get_aorc_config({})
        wrapper = AORCWrapper(DummyEncoder(), 16, disabled)
        baseline_out = wrapper(torch.randn(2, 10))
        self.assertIsInstance(baseline_out, tuple)

        enabled = get_aorc_config({
            'enable_oam': True,
            'enable_orc': True,
            'enable_caa': True,
        })
        wrapper = AORCWrapper(DummyEncoder(), 16, enabled).eval()
        with torch.no_grad():
            out = wrapper(torch.randn(2, 10))
        self.assertIn('embedding', out)
        self.assertEqual(out['embedding'].shape, (2, 16))


if __name__ == '__main__':
    unittest.main()
