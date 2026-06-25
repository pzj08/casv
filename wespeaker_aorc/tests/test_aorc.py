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
from wespeaker.losses.aorc_losses import CrossAgeAggregationLossV2
from wespeaker.losses.aorc_losses import OrdinalAgeLoss
from wespeaker.losses.aorc_losses import OrdinalPrototypeLoss
from wespeaker.losses.aorc_losses import SoftProxyMatchingLoss
from wespeaker.losses.aorc_losses import SpeakerConditionedDirectionLoss
from wespeaker.bin.train import _load_age_labels
from wespeaker.dataset import processor
from wespeaker.models.aorc_modules import AORCWrapper
from wespeaker.models.aorc_modules import AgeResidualCompensation
from wespeaker.models.aorc_modules import OrdinalAgeHead
from wespeaker.models.aorc_modules import _all_gather_no_grad
from wespeaker.models.aorc_modules import _all_gather_with_local_grad
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
            SoftProxyMatchingLoss(groups)(z_age, prototypes, age_group),
            CrossAgeAggregationLossV2(groups)(torch.randn(batch, dim),
                                             speakers, age_group),
        ]
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))

    def test_ordinal_age_loss_ignore_and_backward(self):
        groups = 4
        loss_fn = OrdinalAgeLoss(groups)
        logits = torch.randn(5, groups - 1, requires_grad=True)
        mixed = torch.tensor([0, 1, -1, 2, 3])
        loss = loss_fn(logits, mixed)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        ignored = loss_fn(torch.randn(3, groups - 1), torch.full((3,), -1))
        self.assertEqual(ignored.item(), 0.0)

    def test_soft_proxy_matching_loss_stats_and_grad(self):
        groups = 5
        z_age = torch.randn(8, 6, requires_grad=True)
        prototypes = torch.randn(groups, 6, requires_grad=True)
        age_group = torch.tensor([0, 1, 2, 3, 4, -1, 1, 2])
        loss, stats = SoftProxyMatchingLoss(groups, weight_type="linear")(
            z_age, prototypes, age_group, return_stats=True)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['proxy_valid_count'].item(), 7)
        loss.backward()
        self.assertIsNotNone(prototypes.grad)
        self.assertGreater(prototypes.grad.abs().sum().item(), 0.0)

        ignored, ignored_stats = SoftProxyMatchingLoss(groups)(
            z_age, prototypes, torch.full((8,), -1), return_stats=True)
        self.assertEqual(ignored.item(), 0.0)
        self.assertEqual(ignored_stats['proxy_valid_count'].item(), 0.0)

        one_group = SoftProxyMatchingLoss(1)(
            torch.randn(3, 4), torch.randn(1, 4), torch.zeros(3).long())
        self.assertTrue(torch.isfinite(one_group))

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

    def test_direction_loss_diagnostics(self):
        groups = 4
        speakers = torch.tensor([0, 0, 0, 1, 1])
        age_group = torch.tensor([0, 2, 3, 1, 1])
        z = torch.randn(5, 6)
        prototypes = torch.randn(groups, 6)
        loss, stats = SpeakerConditionedDirectionLoss(
            groups, max_pairs=1)(z, prototypes, speakers, age_group,
                                 return_stats=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['dir_active'].item(), 1.0)
        self.assertEqual(stats['dir_num_pairs_after_subsample'].item(), 1.0)
        self.assertGreater(stats['dir_num_pairs_before_subsample'].item(), 1.0)

        no_pair, no_stats = SpeakerConditionedDirectionLoss(groups)(
            z, prototypes, torch.arange(5), age_group, return_stats=True)
        self.assertEqual(no_pair.item(), 0.0)
        self.assertEqual(no_stats['dir_active'].item(), 0.0)

        same_age, same_stats = SpeakerConditionedDirectionLoss(groups)(
            z, prototypes, speakers, torch.ones(5).long(), return_stats=True)
        self.assertEqual(same_age.item(), 0.0)
        self.assertEqual(same_stats['dir_num_pairs'].item(), 0.0)

        ignored, ignored_stats = SpeakerConditionedDirectionLoss(groups)(
            z, prototypes, speakers, torch.full((5,), -1), return_stats=True)
        self.assertEqual(ignored.item(), 0.0)
        self.assertEqual(ignored_stats['dir_active'].item(), 0.0)

    def test_cross_age_aggregation_v2_modes(self):
        groups = 5
        z = torch.randn(6, 8)
        speakers = torch.tensor([0, 0, 0, 1, 1, 2])
        age_group = torch.tensor([0, 0, 3, 1, 4, 2])

        all_loss, all_stats = CrossAgeAggregationLossV2(
            groups, mode="all_same_speaker")(z, speakers, age_group,
                                             return_stats=True)
        self.assertTrue(torch.isfinite(all_loss))
        self.assertGreater(all_stats['caa_num_positive_pairs'].item(), 0)

        cross_loss, cross_stats = CrossAgeAggregationLossV2(
            groups, mode="cross_age_only", min_gap=1)(z, speakers, age_group,
                                                      return_stats=True)
        self.assertTrue(torch.isfinite(cross_loss))
        self.assertEqual(cross_stats['caa_active'].item(), 1.0)
        self.assertGreater(cross_stats['caa_num_cross_age_positive_pairs'].item(),
                           0)

        large_loss, large_stats = CrossAgeAggregationLossV2(
            groups, mode="large_gap_only", min_gap=3, eta_max=1.1)(
                z, speakers, age_group, return_stats=True)
        self.assertTrue(torch.isfinite(large_loss))
        self.assertGreater(large_loss.item(), 0.0)
        self.assertLessEqual(large_stats['caa_eta_max'].item(), 1.1 + 1e-6)

        disabled, disabled_stats = CrossAgeAggregationLossV2(
            groups, mode="disabled")(z, speakers, age_group, return_stats=True)
        self.assertEqual(disabled.item(), 0.0)
        self.assertEqual(disabled_stats['caa_active'].item(), 0.0)

        same_age = torch.tensor([1, 1, 1, 2, 2, 3])
        no_cross, no_cross_stats = CrossAgeAggregationLossV2(
            groups, mode="cross_age_only")(z, speakers, same_age,
                                           return_stats=True)
        self.assertEqual(no_cross.item(), 0.0)
        self.assertEqual(no_cross_stats['caa_active'].item(), 0.0)

        no_pos, no_pos_stats = CrossAgeAggregationLossV2(groups)(
            z, torch.arange(6), age_group, return_stats=True)
        self.assertEqual(no_pos.item(), 0.0)
        self.assertEqual(no_pos_stats['caa_num_positive_pairs'].item(), 0.0)

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

    def test_wrapper_losses_and_diagnostics(self):
        enabled = get_aorc_config({
            'enable_oam': True,
            'enable_orc': True,
            'enable_caa': True,
            'lambda_caa': 0.005,
        })
        wrapper = AORCWrapper(DummyEncoder(), 16, enabled)
        out = wrapper(torch.randn(6, 10))
        self.assertEqual(out['raw_embedding'].shape, out['embedding'].shape)
        self.assertIn('age_distribution', out)
        self.assertIn('age_prototypes', out)
        speakers = torch.tensor([0, 0, 0, 1, 1, 2])
        age_group = torch.tensor([0, 2, 4, 1, 3, -1])
        losses = wrapper.compute_aorc_losses(out, speakers, age_group, epoch=1)
        self.assertIn('loss_caa', losses)
        self.assertIn('stat_caa_active', losses)
        self.assertIn('stat_dir_active', losses)
        self.assertTrue(torch.isfinite(losses['loss_oam']))

        disabled_caa = get_aorc_config({
            'enable_oam': True,
            'enable_orc': True,
            'enable_caa': False,
        })
        wrapper_no_caa = AORCWrapper(DummyEncoder(), 16, disabled_caa)
        out_no_caa = wrapper_no_caa(torch.randn(4, 10))
        losses_no_caa = wrapper_no_caa.compute_aorc_losses(
            out_no_caa, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2, 1, 3]))
        self.assertEqual(losses_no_caa['loss_caa'].item(), 0.0)

        legacy = get_aorc_config({
            'enable_oam': True,
            'proxy_loss_type': 'legacy',
        })
        wrapper_legacy = AORCWrapper(DummyEncoder(), 16, legacy)
        out_legacy = wrapper_legacy(torch.randn(4, 10))
        legacy_losses = wrapper_legacy.compute_aorc_losses(
            out_legacy, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 2, 3]))
        self.assertTrue(torch.isfinite(legacy_losses['loss_proxy']))

    def test_detached_residual_and_oam_gradients(self):
        config = get_aorc_config({
            'enable_oam': True,
            'enable_orc': True,
            'detach_age_prob_for_residual': True,
        })
        wrapper = AORCWrapper(DummyEncoder(), 16, config)
        out = wrapper(torch.randn(4, 10))
        out['age_distribution'].retain_grad()
        out['embedding'].sum().backward(retain_graph=True)
        self.assertTrue(out['age_distribution'].grad is None
                        or out['age_distribution'].grad.abs().sum().item() == 0.0)

        wrapper.zero_grad()
        losses = wrapper.compute_aorc_losses(
            out, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2, 1, 3]))
        losses['loss_oam'].backward()
        age_grad = sum(
            p.grad.abs().sum().item() for p in wrapper.age_head.parameters()
            if p.grad is not None)
        self.assertGreater(age_grad, 0.0)

    def test_orc_zero_lambda_warning(self):
        with self.assertWarns(RuntimeWarning):
            AORCWrapper(
                DummyEncoder(), 16,
                get_aorc_config({
                    'enable_oam': True,
                    'enable_orc': True,
                    'lambda_oam': 0.0,
                }))

    def test_non_ddp_gather_helpers(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertIs(_all_gather_no_grad(x), x)
        gathered = _all_gather_with_local_grad(x)
        self.assertIs(gathered, x)
        gathered.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_minimal_optimizer_step_updates_aorc_params(self):
        config = get_aorc_config({
            'enable_oam': True,
            'enable_orc': True,
            'enable_caa': True,
        })
        wrapper = AORCWrapper(DummyEncoder(), 16, config)
        optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.1)
        out = wrapper(torch.randn(6, 10))
        losses = wrapper.compute_aorc_losses(
            out, torch.tensor([0, 0, 0, 1, 1, 2]),
            torch.tensor([0, 2, 4, 1, 3, 2]))
        total = losses['loss_oam'] + losses['loss_caa'] + losses['loss_smooth']
        optimizer.zero_grad()
        total.backward()
        self.assertIsNotNone(wrapper.age_head.prototypes.grad)
        self.assertGreater(wrapper.age_head.prototypes.grad.abs().sum().item(),
                           0.0)
        self.assertIsNotNone(wrapper.residual.residual_basis.grad)
        before = wrapper.age_head.prototypes.detach().clone()
        optimizer.step()
        self.assertGreater((wrapper.age_head.prototypes.detach() - before).abs()
                           .sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
