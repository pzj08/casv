import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wespeaker.models.cvac_waveform_modules import (  # noqa: E402
    DifferentiableLogMelFrontend, EmbeddingAgeHead,
    MultiResolutionSTFTLoss, WaveformCVACLoss,
    WaveformCounterfactualAgingGenerator)
from wespeaker.models.resnet import BasicBlock, ResNet  # noqa: E402


def _backward(loss):
    loss.backward()


def test_embedding_age_head():
    b, d, k = 4, 256, 7
    x = torch.randn(b, d, requires_grad=True)
    labels = torch.tensor([0, 1, 2, -1])
    head = EmbeddingAgeHead(d, k, ignore_age_index=-1)
    out = head(x)
    assert out['age_logits'].shape == (b, k)
    assert torch.allclose(out['age_posterior'].sum(-1), torch.ones(b))
    assert torch.all(out['age_uncertainty'] >= 0)
    assert torch.all(out['age_uncertainty'] <= 1)
    loss = head.loss(out['age_logits'], labels)
    assert torch.isfinite(loss)
    _backward(loss)


def test_waveform_generator_shape():
    b, t, d, k = 4, 4000, 256, 7
    x = torch.randn(b, t).clamp(-1, 1).requires_grad_()
    z = F.normalize(torch.randn(b, d), dim=-1)
    q_src = torch.softmax(torch.randn(b, k), dim=-1)
    q_tgt = torch.softmax(torch.randn(b, k), dim=-1)
    gen = WaveformCounterfactualAgingGenerator(d,
                                               k,
                                               hidden_channels=16,
                                               condition_dim=32,
                                               num_layers=2,
                                               dilations=[1, 2])
    out = gen(x, z, q_src, q_tgt)
    assert out['waveform'].shape == (b, t)
    assert out['residual'].shape == (b, t)
    assert out['gate'].shape == (b, 1)
    assert torch.isfinite(out['waveform']).all()
    assert out['waveform'].max() <= 1.0
    assert out['waveform'].min() >= -1.0
    _backward(out['waveform'].mean() + out['residual'].mean())


def test_multi_resolution_stft_loss():
    loss_fn = MultiResolutionSTFTLoss((64, 128), (16, 32), (64, 128))
    x = torch.randn(2, 1024, requires_grad=True)
    same = loss_fn(x, x)
    diff = loss_fn(x, x * 0.5)
    assert torch.isfinite(same)
    assert torch.isfinite(diff)
    assert same.item() < 1.0e-5
    assert diff.item() > 0
    _backward(diff)


def test_differentiable_logmel_frontend():
    frontend = DifferentiableLogMelFrontend(num_mel_bins=80,
                                            n_fft=128,
                                            hop_length=40,
                                            win_length=80)
    x = torch.randn(2, 1600, requires_grad=True)
    y = frontend(x)
    assert y.shape[0] == 2
    assert y.shape[-1] == 80
    assert torch.isfinite(y).all()
    _backward(y.mean())


class _ToyEmbed(nn.Module):

    def __init__(self, t, d):
        super().__init__()
        self.proj = nn.Linear(t, d)

    def forward(self, wav):
        return F.normalize(self.proj(wav), dim=-1)


def _small_cvac_loss():
    return WaveformCVACLoss(lambda_cf_align=0.003,
                            lambda_id=0.003,
                            lambda_age=0.001,
                            lambda_cycle=0.001,
                            lambda_neg=0.001,
                            lambda_mrstft=0.002,
                            lambda_energy=0.0005,
                            lambda_residual=0.0005,
                            max_pairs=4,
                            stft_fft_sizes=[64],
                            stft_hop_sizes=[16],
                            stft_win_lengths=[64])


def test_waveform_cvac_loss_no_pair():
    b, t, d, k = 4, 1024, 32, 7
    wave = torch.randn(b, t)
    emb = F.normalize(torch.randn(b, d), dim=-1).requires_grad_()
    q = torch.softmax(torch.randn(b, k), dim=-1)
    u = torch.zeros(b, 1)
    speakers = torch.arange(b)
    ages = torch.zeros(b, dtype=torch.long)
    gen = WaveformCounterfactualAgingGenerator(d,
                                               k,
                                               hidden_channels=8,
                                               condition_dim=16,
                                               num_layers=1,
                                               dilations=[1])
    out = _small_cvac_loss()(wave, emb, q, u, speakers, ages, gen,
                             _ToyEmbed(t, d))
    assert out['wavcvac_pair_count'].item() == 0
    for key, value in out.items():
        assert torch.isfinite(value)
        assert value.item() == 0


def test_waveform_cvac_loss_valid_pair():
    b, t, d, k = 4, 1024, 32, 7
    wave = torch.randn(b, t).clamp(-1, 1).requires_grad_()
    emb = F.normalize(torch.randn(b, d), dim=-1).requires_grad_()
    q = torch.softmax(torch.randn(b, k), dim=-1)
    u = torch.zeros(b, 1)
    speakers = torch.tensor([0, 0, 1, 2])
    ages = torch.tensor([1, 3, 2, 4])
    gen = WaveformCounterfactualAgingGenerator(d,
                                               k,
                                               hidden_channels=8,
                                               condition_dim=16,
                                               num_layers=1,
                                               dilations=[1])
    age_head = EmbeddingAgeHead(d, k, age_hidden_dim=16)
    out = _small_cvac_loss()(wave, emb, q, u, speakers, ages, gen,
                             _ToyEmbed(t, d), age_head)
    assert out['wavcvac_pair_count'].item() > 0
    assert torch.isfinite(out['loss_wavcvac_total'])
    _backward(out['loss_wavcvac_total'])


def _small_model(cvac_enabled):
    cvac_args = {
        'enabled': cvac_enabled,
        'num_age_groups': 7,
        'age_hidden_dim': 16,
        'condition_dim': 16,
        'hidden_channels': 8,
        'num_layers': 1,
        'dilations': [1],
        'mel_bins': 80,
        'mel_fft_size': 128,
        'mel_hop_size': 40,
        'mel_win_length': 80,
        'stft_fft_sizes': [64],
        'stft_hop_sizes': [16],
        'stft_win_lengths': [64],
        'max_pairs': 4,
    }
    return ResNet(BasicBlock, [1, 1, 1, 1],
                  m_channels=8,
                  feat_dim=80,
                  embed_dim=32,
                  cvac_args=cvac_args)


def test_model_smoke_enabled():
    model = _small_model(True)
    model.train()
    feats = torch.randn(4, 40, 80)
    wave = torch.randn(4, 1600).clamp(-1, 1)
    speakers = torch.tensor([0, 0, 1, 2])
    ages = torch.tensor([1, 3, 2, 4])
    emb = model(feats)[-1]
    age_out = model.cvac_age_head(emb)
    out = model.cvac_loss(wave, emb, age_out['age_posterior'],
                          age_out['age_uncertainty'], speakers, ages,
                          model.cvac_generator, model.forward_waveform_for_cvac,
                          model.cvac_age_head)
    loss = emb.pow(2).mean() + model.cvac_age_head.loss(
        age_out['age_logits'], ages) + out['loss_wavcvac_total']
    assert 'loss_wavcvac_total' in out
    assert torch.isfinite(loss)
    _backward(loss)


def test_disabled_behavior():
    model = _small_model(False)
    feats = torch.randn(2, 40, 80)
    out = model(feats)
    assert not model.cvac_enabled
    assert model.cvac_loss is None
    assert out[-1].shape == (2, 32)


def test_missing_waveform_safety():
    model = _small_model(True)
    feats = torch.randn(2, 40, 80)
    emb = model(feats)[-1]
    age_group = torch.tensor([1, 2])
    age_out = model.cvac_age_head(emb)
    age_loss = model.cvac_age_head.loss(age_out['age_logits'], age_group)
    zero_cvac = emb.sum() * 0.0
    missing_waveform = torch.tensor(1.0)
    assert torch.isfinite(age_loss + zero_cvac)
    assert missing_waveform.item() == 1.0


if __name__ == '__main__':
    tests = [
        test_embedding_age_head,
        test_waveform_generator_shape,
        test_multi_resolution_stft_loss,
        test_differentiable_logmel_frontend,
        test_waveform_cvac_loss_no_pair,
        test_waveform_cvac_loss_valid_pair,
        test_model_smoke_enabled,
        test_disabled_behavior,
        test_missing_waveform_safety,
    ]
    for test in tests:
        test()
