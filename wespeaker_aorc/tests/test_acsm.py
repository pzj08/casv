import os
import sys
import tempfile
import types
import unittest

import torch
import torch.nn.functional as F
import yaml

silero_vad = types.ModuleType('silero_vad')
silero_vad.load_silero_vad = lambda *args, **kwargs: None
silero_vad.read_audio = lambda *args, **kwargs: None
silero_vad.get_speech_timestamps = lambda *args, **kwargs: []
sys.modules.setdefault('silero_vad', silero_vad)

from wespeaker.bin.train import _validate_acsm_age_label_config
from wespeaker.models.acsm_modules import AgeFiLM2d
from wespeaker.models.acsm_modules import OrderedAgeCanonicalizer
from wespeaker.models.acsm_modules import PathConsistencyLoss
from wespeaker.models.acsm_modules import Stage2AgeObserver
from wespeaker.models.acsm_modules import get_acsm_config
from wespeaker.models.aorc_modules import AORCWrapper, get_aorc_config
from wespeaker.models.resnet import ResNet34, ResNet34_ACSM
from wespeaker.models.resnet import ResNet34_ParamMatch
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


class ACSMTest(unittest.TestCase):

    def test_get_speaker_model_builds_resnet34_acsm_from_main_config(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(
            repo_root, 'examples', 'voxceleb', 'v2', 'conf',
            'resnet34_acsm_main.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_cls = get_speaker_model(config['model'])
        self.assertIs(model_cls, ResNet34_ACSM)

        config['model_args']['acsm_args'] = get_acsm_config(config)
        model = model_cls(**config['model_args'])
        model.eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 200, 80))
        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs['embedding'].shape, (2, 256))
        self.assertEqual(outputs['age_posterior'].shape, (2, 7))
        self.assertTrue(torch.isfinite(outputs['embedding']).all())

    def test_path_ablation_configs_keep_main_hparams(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expected = {
            'resnet34_acsm_path000.yaml': 0.0,
            'resnet34_acsm_path001.yaml': 0.01,
            'resnet34_acsm_path002.yaml': 0.02,
        }
        for filename, lambda_path in expected.items():
            path = os.path.join(repo_root, 'examples', 'voxceleb', 'v2',
                                'conf', filename)
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            acsm = config['model_args']['acsm_args']
            self.assertEqual(config['model'], 'ResNet34_ACSM')
            self.assertEqual(acsm['losses']['lambda_path'], lambda_path)
            self.assertEqual(acsm['losses']['lambda_age'], 0.10)
            self.assertEqual(acsm['losses']['lambda_consistency'], 0.03)
            self.assertEqual(acsm['losses']['lambda_smooth'], 1.0e-4)
            self.assertEqual(acsm['losses']['ramp_epoch'], 3)
            self.assertEqual(acsm['canonicalizer']['gate_max'], 0.40)
            self.assertEqual(acsm['canonicalizer']['canonical_scale'], 0.10)

    def test_age_film_identity_initialization(self):
        film = AgeFiLM2d(8, 4, film_scale=0.05)
        h = torch.randn(3, 8, 5, 7)
        q_age = F.softmax(torch.randn(3, 4), dim=-1)
        out = film(h, q_age)
        self.assertEqual(out.shape, h.shape)
        self.assertTrue(torch.allclose(out, h, atol=1e-6))
        self.assertFalse(torch.isnan(out).any())

        disabled = AgeFiLM2d(8, 4, enabled=False)
        self.assertTrue(torch.equal(disabled(h, q_age), h))

    def test_stage2_age_observer_outputs_and_loss(self):
        observer = Stage2AgeObserver(16, 5, age_emb_dim=8)
        h2 = torch.randn(4, 16, 10, 20)
        out = observer(h2)
        self.assertEqual(out['age_posterior'].shape, (4, 5))
        self.assertEqual(out['rank_logits'].shape, (4, 4))
        self.assertTrue(
            torch.allclose(out['age_posterior'].sum(dim=1),
                           torch.ones(4),
                           atol=1e-5))
        self.assertTrue(torch.isfinite(out['age_posterior']).all())
        self.assertTrue((out['age_posterior'] >= 0.0).all())

        age_group = torch.tensor([0, 1, 3, -1])
        loss = observer.ordinal_loss(out['rank_logits'], age_group)
        self.assertTrue(torch.isfinite(loss))
        ignored = observer.ordinal_loss(out['rank_logits'],
                                        torch.full((4,), -1))
        self.assertEqual(ignored.item(), 0.0)

    def test_ordered_age_canonicalizer(self):
        canon = OrderedAgeCanonicalizer(5, 12, reference_age_group=2)
        e_obs = torch.randn(4, 12)
        q_age = F.one_hot(torch.tensor([2, 0, 4, 1]), num_classes=5).float()
        out = canon(e_obs, q_age)
        self.assertEqual(out['embedding'].shape, e_obs.shape)
        self.assertEqual(out['gate'].shape, (4, 1))
        self.assertGreaterEqual(out['gate'].min().item(), -1e-6)
        self.assertLessEqual(out['gate'].max().item(), 0.5 + 1e-6)
        self.assertTrue(torch.isfinite(out['transition_smooth_loss']))
        self.assertTrue(
            torch.allclose(out['embedding'].norm(dim=-1),
                           torch.ones(4),
                           atol=1e-5))
        self.assertTrue(
            torch.allclose(out['canonical_residual'][0],
                           torch.zeros(12),
                           atol=1e-7))
        self.assertFalse(torch.isnan(out['embedding']).any())

        identity = OrderedAgeCanonicalizer(5,
                                           12,
                                           reference_age_group=2,
                                           canonical_scale=0.0)
        identity_out = identity(e_obs, q_age)
        self.assertTrue(
            torch.allclose(identity_out['embedding'],
                           F.normalize(e_obs, dim=-1),
                           atol=1e-6))

    def test_acsm_resnet_forward_without_age_group(self):
        model = ResNet34_ACSM(feat_dim=80,
                              embed_dim=16,
                              acsm_args={
                                  'num_age_groups': 4,
                                  'reference_age_group': 1,
                                  'age_emb_dim': 8,
                              })
        out = model(torch.randn(2, 64, 80))
        for key in [
                'embedding', 'raw_embedding', 'age_posterior', 'rank_logits',
                'gate', 'uncertainty'
        ]:
            self.assertIn(key, out)
        self.assertEqual(out['embedding'].shape, (2, 16))
        self.assertEqual(out['raw_embedding'].shape, (2, 16))
        self.assertEqual(out['age_posterior'].shape, (2, 4))
        self.assertTrue(torch.isfinite(out['embedding']).all())
        self.assertTrue(torch.isfinite(out['raw_embedding']).all())
        self.assertTrue(torch.isfinite(out['age_posterior']).all())

    def test_acsm_losses_backward(self):
        torch.manual_seed(0)
        model = ResNet34_ACSM(feat_dim=80,
                              embed_dim=16,
                              acsm_args={
                                  'num_age_groups': 4,
                                  'reference_age_group': 1,
                                  'age_emb_dim': 8,
                                  'losses': {
                                      'lambda_age': 0.1,
                                      'lambda_consistency': 0.02,
                                      'lambda_smooth': 1.0e-4,
                                      'lambda_path': 0.1,
                                      'ramp_epoch': 0,
                                  },
                              })
        projection = torch.nn.Linear(16, 3)
        outputs = model(torch.randn(4, 80, 80))
        speakers = torch.tensor([0, 0, 1, 1])
        age_group = torch.tensor([0, 2, 1, 3])
        logits = projection(outputs['embedding'])
        spk_loss = F.cross_entropy(logits, speakers)
        extra = model.compute_acsm_losses(outputs, speakers, age_group, epoch=1)
        loss = spk_loss + extra['loss_acsm_total']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.age_observer.score.weight.grad)
        self.assertIsNotNone(model.age_film3.gamma.weight.grad)
        self.assertIsNotNone(model.canonicalizer.gate_mlp[-1].weight.grad)
        self.assertIsNotNone(model.canonicalizer.adjacent_transitions.grad)
        self.assertTrue(
            torch.isfinite(model.age_observer.score.weight.grad).all())

        ignored = model.compute_acsm_losses(
            outputs, speakers, torch.full((4,), -1), epoch=1)
        self.assertTrue(torch.isfinite(ignored['loss_age']))
        self.assertEqual(ignored['loss_age'].item(), 0.0)

    def test_path_loss_pair_count(self):
        model = ResNet34_ACSM(feat_dim=80,
                              embed_dim=16,
                              acsm_args={
                                  'num_age_groups': 4,
                                  'reference_age_group': 1,
                                  'age_emb_dim': 8,
                                  'losses': {
                                      'lambda_path': 0.01,
                                      'ramp_epoch': 0,
                                  },
                              })
        outputs = model(torch.randn(4, 80, 80))
        speakers = torch.tensor([0, 0, 1, 2])
        age_group = torch.tensor([0, 2, 1, 3])
        losses = model.compute_acsm_losses(outputs, speakers, age_group)
        self.assertGreater(losses['path_valid_pair_count'].item(), 0.0)
        self.assertTrue(torch.isfinite(losses['loss_path']))

        no_pair = model.compute_acsm_losses(outputs, torch.arange(4),
                                            age_group)
        self.assertEqual(no_pair['path_valid_pair_count'].item(), 0.0)
        self.assertEqual(no_pair['loss_path'].item(), 0.0)

    def test_path_consistency_loss_valid_pairs_and_ignore_age(self):
        loss_fn = PathConsistencyLoss(ignore_age_index=-1)
        embeddings = F.normalize(torch.randn(5, 8), dim=-1)
        speakers = torch.tensor([0, 0, 0, 1, 1])
        age_group = torch.tensor([0, 2, -1, 1, 1])

        pairs = loss_fn.valid_pair_indices(speakers, age_group)
        self.assertEqual(pairs.shape[0], 1)
        self.assertEqual(loss_fn.valid_pair_count(embeddings, speakers,
                                                  age_group).item(), 1.0)
        loss = loss_fn(embeddings, speakers, age_group)
        self.assertTrue(torch.isfinite(loss))

        no_pair_speakers = torch.arange(5)
        no_pair = loss_fn(embeddings, no_pair_speakers, age_group)
        self.assertEqual(no_pair.item(), 0.0)
        ignored = loss_fn(embeddings, speakers, torch.full((5,), -1))
        self.assertEqual(ignored.item(), 0.0)

    def test_missing_age_label_config(self):
        conf = get_acsm_config({
            'model': 'ResNet34_ACSM',
            'model_args': {
                'acsm_args': {
                    'age_label_file': None,
                    'losses': {
                        'lambda_age': 0.1,
                        'lambda_path': 0.0,
                    },
                }
            }
        })
        with self.assertRaises(ValueError):
            _validate_acsm_age_label_config(conf)

        conf['losses']['lambda_age'] = 0.0
        _validate_acsm_age_label_config(conf)

    def test_baseline_and_aorc_still_construct(self):
        baseline = ResNet34(feat_dim=80, embed_dim=16)
        base_out = baseline(torch.randn(2, 64, 80))
        self.assertIsInstance(base_out, tuple)
        self.assertEqual(base_out[-1].shape, (2, 16))

        aorc_conf = get_aorc_config({
            'aorc_args': {
                'enable_oam': True,
                'num_age_groups': 4,
                'age_emb_dim': 8,
            }
        })
        wrapped = AORCWrapper(ResNet34(feat_dim=80, embed_dim=16), 16,
                              aorc_conf)
        out = wrapped(torch.randn(2, 64, 80))
        self.assertIn('embedding', out)
        self.assertEqual(out['embedding'].shape, (2, 16))

    def test_resnet34_parammatch_constructs_without_age_semantics(self):
        model_cls = get_speaker_model('ResNet34_ParamMatch')
        self.assertIs(model_cls, ResNet34_ParamMatch)
        model = model_cls(feat_dim=80,
                          embed_dim=16,
                          param_match_args={
                              'bottleneck_dim': 8,
                              'residual_scale': 0.1,
                          })
        out = model(torch.randn(2, 200, 80))
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[-1].shape, (2, 16))
        self.assertTrue(torch.isfinite(out[-1]).all())
        self.assertFalse(hasattr(model, 'age_observer'))
        self.assertFalse(hasattr(model, 'canonicalizer'))

    def test_resnet34_parammatch_parameter_count_matches_acsm(self):
        baseline = ResNet34(feat_dim=80, embed_dim=256)
        acsm = ResNet34_ACSM(feat_dim=80, embed_dim=256)
        parammatch = ResNet34_ParamMatch(feat_dim=80, embed_dim=256)
        base_n = sum(p.numel() for p in baseline.parameters())
        acsm_extra = sum(p.numel() for p in acsm.parameters()) - base_n
        pm_extra = sum(p.numel() for p in parammatch.parameters()) - base_n
        self.assertGreater(pm_extra, 0)
        self.assertGreater(sum(p.numel() for p in parammatch.parameters()),
                           base_n)
        self.assertLess(abs(pm_extra - acsm_extra), acsm_extra * 0.25)

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, 'examples', 'voxceleb', 'v2',
                                   'conf', 'resnet34_parammatch.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.assertEqual(config['model'], 'ResNet34_ParamMatch')
        self.assertNotIn('acsm_args', config['model_args'])
        self.assertNotIn('age_label_file', str(config['model_args']))

    def test_extraction_style_forward_needs_no_age_label(self):
        model = ResNet34_ACSM(feat_dim=80,
                              embed_dim=16,
                              acsm_args={
                                  'num_age_groups': 4,
                                  'reference_age_group': 1,
                                  'age_emb_dim': 8,
                              })
        model.eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 200, 80))
        self.assertIsInstance(outputs, dict)
        self.assertIn('age_posterior', outputs)
        self.assertTrue(torch.isfinite(outputs['age_posterior']).all())

        embeds = outputs['embedding'] if isinstance(outputs,
                                                    dict) else outputs[-1]
        self.assertEqual(embeds.shape, (2, 16))
        self.assertTrue(torch.isfinite(embeds).all())

    def test_resnet34_checkpoint_partial_loads_into_acsm(self):
        baseline = ResNet34(feat_dim=80, embed_dim=16)
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.save(baseline.state_dict(), tmp.name)
            acsm = ResNet34_ACSM(feat_dim=80,
                                 embed_dim=16,
                                 acsm_args={
                                     'num_age_groups': 4,
                                     'reference_age_group': 1,
                                     'age_emb_dim': 8,
                                 })
            report = load_checkpoint(acsm,
                                     tmp.name,
                                     strict=False,
                                     allow_acsm_partial=True)
            self.assertGreater(report['missing_acsm_key_count'], 0)
            self.assertTrue(
                torch.allclose(acsm.conv1.weight, baseline.conv1.weight))

            strict_acsm = ResNet34_ACSM(feat_dim=80,
                                        embed_dim=16,
                                        acsm_args={
                                            'num_age_groups': 4,
                                            'reference_age_group': 1,
                                            'age_emb_dim': 8,
                                        })
            with self.assertRaises(RuntimeError):
                load_checkpoint(strict_acsm, tmp.name, strict=True)


if __name__ == '__main__':
    unittest.main()
