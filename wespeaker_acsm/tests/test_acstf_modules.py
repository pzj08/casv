import unittest
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wespeaker.models.acsm_modules import AgeTrajectoryLoss
from wespeaker.models.acsm_modules import AgeTrajectoryTransport
from wespeaker.models.acsm_modules import OrderedAgeCanonicalizer
from wespeaker.models.resnet import ResNet34_ACSM


class ACSTFModuleTest(unittest.TestCase):

    def test_expected_transport_residual(self):
        canonicalizer = OrderedAgeCanonicalizer(num_age_groups=7,
                                                embedding_dim=256,
                                                reference_age_group=3)
        q_src = F.one_hot(torch.tensor([0, 2, 4, 6]),
                          num_classes=7).float()
        q_tgt = F.one_hot(torch.tensor([0, 3, 1, 6]),
                          num_classes=7).float()
        same = canonicalizer.expected_transport_residual(q_src, q_src)
        residual = canonicalizer.expected_transport_residual(q_src, q_tgt)
        self.assertEqual(residual.shape, (4, 256))
        self.assertTrue(torch.allclose(same, torch.zeros_like(same)))
        self.assertTrue(torch.isfinite(residual).all())

    def test_age_trajectory_transport_forward(self):
        torch.manual_seed(1)
        canonicalizer = OrderedAgeCanonicalizer(7, 256, reference_age_group=3)
        transport = AgeTrajectoryTransport(7, 256)
        e_src = F.normalize(torch.randn(8, 256), dim=-1)
        src_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 1])
        tgt_idx = torch.tensor([1, 2, 3, 4, 5, 6, 0, 1])
        q_src = F.one_hot(src_idx, num_classes=7).float()
        q_tgt = F.one_hot(tgt_idx, num_classes=7).float()
        out = transport(e_src, q_src, q_tgt, canonicalizer)
        self.assertEqual(out['embedding'].shape, (8, 256))
        self.assertEqual(out['transport_gate'].shape, (8, 1))
        self.assertTrue(
            torch.allclose(out['embedding'].norm(dim=-1),
                           torch.ones(8),
                           atol=1e-5))
        self.assertTrue(torch.isfinite(out['transport_residual_norm']).all())

    def test_age_trajectory_loss_no_pair(self):
        canonicalizer = OrderedAgeCanonicalizer(7, 256, reference_age_group=3)
        transport = AgeTrajectoryTransport(7, 256)
        loss_fn = AgeTrajectoryLoss(ignore_age_index=-1)
        embeddings = F.normalize(torch.randn(4, 256), dim=-1)
        q_age = F.one_hot(torch.tensor([1, 1, 1, 1]),
                          num_classes=7).float()
        speakers = torch.arange(4)
        age_group = torch.tensor([1, 1, 1, 1])
        losses = loss_fn(embeddings, q_age, speakers, age_group, transport,
                         canonicalizer)
        self.assertEqual(losses['loss_transport'].item(), 0.0)
        self.assertEqual(losses['loss_transport_cycle'].item(), 0.0)
        self.assertEqual(losses['loss_transport_identity'].item(), 0.0)
        self.assertEqual(losses['transport_pair_count'].item(), 0.0)

    def test_age_trajectory_loss_valid_pair_backward(self):
        torch.manual_seed(2)
        canonicalizer = OrderedAgeCanonicalizer(7, 256, reference_age_group=3)
        transport = AgeTrajectoryTransport(7, 256)
        loss_fn = AgeTrajectoryLoss(ignore_age_index=-1,
                                    min_age_gap=1,
                                    max_pairs=16,
                                    bidirectional=True)
        embeddings = F.normalize(torch.randn(6, 256), dim=-1)
        embeddings.requires_grad_()
        ages = torch.tensor([0, 2, 1, 3, 4, 6])
        q_age = F.one_hot(ages, num_classes=7).float()
        speakers = torch.tensor([0, 0, 1, 1, 2, 2])
        losses = loss_fn(embeddings, q_age, speakers, ages, transport,
                         canonicalizer)
        total = (losses['loss_transport'] + losses['loss_transport_cycle'] +
                 losses['loss_transport_identity'])
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(losses['transport_pair_count'].item(), 0.0)
        total.backward()
        adj_grad = canonicalizer.adjacent_transitions.grad
        gate_grad = transport.gate_mlp[-1].bias.grad
        self.assertTrue((adj_grad is not None and torch.isfinite(adj_grad).all())
                        or (gate_grad is not None
                            and torch.isfinite(gate_grad).all()))

    def test_resnet34_acsm_acstf_forward_backward_smoke(self):
        torch.manual_seed(3)
        model = ResNet34_ACSM(
            feat_dim=80,
            embed_dim=256,
            acsm_args={
                'num_age_groups': 7,
                'reference_age_group': 3,
                'age_emb_dim': 32,
                'losses': {
                    'lambda_age': 0.1,
                    'lambda_consistency': 0.02,
                    'lambda_smooth': 1.0e-4,
                    'lambda_path': 0.0,
                    'ramp_epoch': 0,
                },
                'trajectory': {
                    'enabled': True,
                    'lambda_transport': 0.01,
                    'lambda_cycle': 0.002,
                    'lambda_identity': 0.002,
                    'use_raw_embedding': True,
                    'detach_target': True,
                    'min_age_gap': 1,
                    'max_pairs': 512,
                    'bidirectional': True,
                    'gate_max': 0.4,
                    'gate_init_bias': -2.0,
                    'loss_type': 'cosine',
                },
            })
        outputs = model(torch.randn(4, 200, 80))
        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs['embedding'].shape, (4, 256))
        self.assertTrue(outputs['acstf_enabled'])
        speakers = torch.tensor([0, 0, 1, 1])
        age_group = torch.tensor([0, 1, 2, 3])
        losses = model.compute_acsm_losses(outputs,
                                           speakers,
                                           age_group,
                                           epoch=1)
        for key in [
                'loss_transport', 'loss_transport_cycle',
                'loss_transport_identity', 'loss_acstf_total',
                'transport_pair_count', 'transport_gate_mean',
                'transport_residual_norm_mean', 'transport_cos_pos_mean'
        ]:
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]).all())
        losses['loss_acsm_total'].backward()

    def test_resnet34_acsm_trajectory_disabled_zero_loss(self):
        torch.manual_seed(4)
        model = ResNet34_ACSM(feat_dim=80,
                              embed_dim=256,
                              acsm_args={
                                  'num_age_groups': 7,
                                  'reference_age_group': 3,
                                  'age_emb_dim': 32,
                                  'trajectory': {
                                      'enabled': False,
                                  },
                              })
        outputs = model(torch.randn(4, 200, 80))
        speakers = torch.tensor([0, 0, 1, 1])
        age_group = torch.tensor([0, 1, 2, 3])
        losses = model.compute_acsm_losses(outputs,
                                           speakers,
                                           age_group,
                                           epoch=1)
        self.assertEqual(outputs['embedding'].shape, (4, 256))
        self.assertFalse(outputs['acstf_enabled'])
        self.assertEqual(losses['loss_acstf_total'].item(), 0.0)
        self.assertEqual(losses['loss_transport'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
