# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from wespeaker.losses.aorc_losses import OrdinalAgeLoss


ACSM_DEFAULTS = {
    'enabled': True,
    'num_age_groups': 7,
    'age_bins': None,
    'age_label_file': None,
    'age_label_type': 'group',
    'ignore_age_index': -1,
    'reference_age_group': 3,
    'age_observer_stage': 'layer2',
    'age_emb_dim': 128,
    'film': {
        'enabled': True,
        'stages': ['layer3', 'layer4'],
        'film_scale': 0.05,
    },
    'canonicalizer': {
        'enabled': True,
        'canonical_scale': 0.1,
        'learnable_canonical_scale': False,
        'gate_max': 0.5,
        'gate_init_bias': -2.0,
        'transition_init_std': 0.005,
    },
    'losses': {
        'lambda_age': 0.1,
        'lambda_consistency': 0.02,
        'lambda_smooth': 1.0e-4,
        'lambda_path': 0.0,
        'ramp_epoch': 2,
    },
    'consistency': {
        'mode': 'embedding',
        'small_age_gap': 1,
        'only_small_age_gap': False,
    },
    'diagnostics': {
        'strict_finite_check': True,
        'log_diagnostics': True,
    },
    'eps': 1.0e-12,
}


def _deep_update(base, updates):
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def get_acsm_config(configs):
    """Return normalized ACSM config from top-level or model_args config."""
    model_args = configs.get('model_args', {})
    user = dict(model_args.get('acsm_args', configs.get('acsm_args', {})) or {})
    conf = _deep_update(ACSM_DEFAULTS, user)
    if conf['age_bins'] is not None:
        conf['num_age_groups'] = len(conf['age_bins']) + 1
    if not (0 <= int(conf['reference_age_group']) < int(conf['num_age_groups'])):
        raise ValueError('reference_age_group must be in [0, num_age_groups)')
    return conf


def acsm_is_enabled(configs):
    """Check whether the config requests the structural ACSM ResNet variant."""
    model_name = configs.get('model', '')
    if model_name not in ('ResNet34_ACSM', 'ACSM_ResNet34'):
        return False
    return bool(get_acsm_config(configs).get('enabled', True))


def _zero_like(x):
    return x.new_zeros(())


def _stat(value, ref):
    if torch.is_tensor(value):
        return value.detach()
    return ref.new_tensor(float(value)).detach()


class AgeFiLM2d(nn.Module):
    """Small age-conditioned FiLM adapter for a 2-D ResNet feature map."""

    def __init__(self,
                 channels,
                 num_age_groups,
                 film_scale=0.05,
                 enabled=True):
        super().__init__()
        self.channels = channels
        self.num_age_groups = num_age_groups
        self.film_scale = float(film_scale)
        self.enabled = bool(enabled)
        self.gamma = nn.Linear(num_age_groups, channels)
        self.beta = nn.Linear(num_age_groups, channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h, q_age):
        if not self.enabled:
            return h
        gamma = self.gamma(q_age).to(dtype=h.dtype).view(h.size(0), -1, 1, 1)
        beta = self.beta(q_age).to(dtype=h.dtype).view(h.size(0), -1, 1, 1)
        return h * (1.0 + self.film_scale * gamma) + self.film_scale * beta


class Stage2AgeObserver(nn.Module):
    """Predict an ordered age posterior from ResNet layer2 features."""

    def __init__(self,
                 in_channels,
                 num_age_groups,
                 age_emb_dim=128,
                 ignore_age_index=-1,
                 eps=1.0e-12):
        super().__init__()
        self.in_channels = in_channels
        self.num_age_groups = num_age_groups
        self.age_emb_dim = age_emb_dim
        self.ignore_age_index = ignore_age_index
        self.eps = eps
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_channels, age_emb_dim)
        self.score = nn.Linear(age_emb_dim, 1)
        self.raw_delta = nn.Parameter(torch.zeros(num_age_groups - 1))
        self.loss_ord = OrdinalAgeLoss(num_age_groups, ignore_age_index)

    def _thresholds(self):
        delta = F.softplus(self.raw_delta) + self.eps
        thresholds = torch.cumsum(delta, dim=0)
        return thresholds - thresholds.mean()

    def _ordinal_distribution(self, rank_logits):
        prob = torch.sigmoid(rank_logits)
        pieces = [1.0 - prob[:, :1]]
        if self.num_age_groups > 2:
            pieces.append(prob[:, :-1] - prob[:, 1:])
        pieces.append(prob[:, -1:])
        q_age = torch.cat(pieces, dim=1)
        q_age = q_age.clamp_min(self.eps)
        return q_age / q_age.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, h2):
        pooled = self.pool(h2).flatten(1)
        age_embedding = F.normalize(self.proj(pooled), dim=-1, eps=self.eps)
        score = self.score(age_embedding)
        rank_logits = score - self._thresholds().view(1, -1)
        age_posterior = self._ordinal_distribution(rank_logits)
        groups = torch.arange(self.num_age_groups,
                              device=h2.device,
                              dtype=age_posterior.dtype)
        age_pred = torch.sum(age_posterior * groups.view(1, -1), dim=-1)
        return {
            'age_embedding': age_embedding,
            'rank_logits': rank_logits,
            'age_posterior': age_posterior,
            'age_pred': age_pred,
        }

    def ordinal_loss(self, rank_logits, age_group):
        return self.loss_ord(rank_logits, age_group)


class OrderedAgeCanonicalizer(nn.Module):
    """Map observed speaker embeddings to a reference-age manifold."""

    def __init__(self,
                 num_age_groups,
                 embedding_dim,
                 reference_age_group,
                 canonical_scale=0.1,
                 learnable_canonical_scale=False,
                 gate_max=0.5,
                 gate_init_bias=-2.0,
                 transition_init_std=0.005,
                 enabled=True,
                 eps=1.0e-12):
        super().__init__()
        self.num_age_groups = num_age_groups
        self.embedding_dim = embedding_dim
        self.reference_age_group = reference_age_group
        self.gate_max = float(gate_max)
        self.enabled = bool(enabled)
        self.eps = eps
        transitions = torch.empty(max(num_age_groups - 1, 0), embedding_dim)
        nn.init.normal_(transitions, mean=0.0, std=float(transition_init_std))
        self.adjacent_transitions = nn.Parameter(transitions)
        scale = torch.tensor(float(canonical_scale))
        if learnable_canonical_scale:
            self.canonical_scale = nn.Parameter(scale)
        else:
            self.register_buffer('canonical_scale', scale)
        gate_hidden = max(1, embedding_dim // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim + num_age_groups + 1, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_init_bias))

    def _paths_to_reference(self):
        paths = self.adjacent_transitions.new_zeros(self.num_age_groups,
                                                   self.embedding_dim)
        r = self.reference_age_group
        for k in range(self.num_age_groups):
            if k < r:
                paths[k] = self.adjacent_transitions[k:r].sum(dim=0)
            elif k > r:
                paths[k] = -self.adjacent_transitions[r:k].sum(dim=0)
        return paths

    def transition_smooth_loss(self):
        if self.adjacent_transitions.size(0) < 2:
            return _zero_like(self.adjacent_transitions)
        diffs = (self.adjacent_transitions[1:] -
                 self.adjacent_transitions[:-1])
        return diffs.pow(2).sum(dim=-1).mean()

    def forward(self, e_obs, q_age):
        paths = self._paths_to_reference()
        canonical_residual = torch.matmul(q_age.to(dtype=e_obs.dtype), paths)
        entropy = -(q_age.clamp_min(self.eps) *
                    q_age.clamp_min(self.eps).log()).sum(dim=-1,
                                                         keepdim=True)
        denom = math.log(max(self.num_age_groups, 2))
        uncertainty = (entropy / denom).clamp(0.0, 1.0).to(dtype=e_obs.dtype)
        gate_input = torch.cat([e_obs, q_age.to(dtype=e_obs.dtype),
                                uncertainty],
                               dim=-1)
        gate = self.gate_max * torch.sigmoid(self.gate_mlp(gate_input))
        gate = gate * (1.0 - uncertainty)
        if self.enabled:
            scale = self.canonical_scale.to(device=e_obs.device,
                                            dtype=e_obs.dtype)
            e_can = e_obs + gate * scale * canonical_residual
        else:
            e_can = e_obs
            canonical_residual = torch.zeros_like(e_obs)
            gate = e_obs.new_zeros(e_obs.size(0), 1)
        e_can = F.normalize(e_can, dim=-1, eps=self.eps)
        return {
            'embedding': e_can,
            'canonical_residual': canonical_residual,
            'gate': gate,
            'uncertainty': uncertainty.squeeze(-1),
            'path_norm': canonical_residual.norm(dim=-1),
            'transition_smooth_loss': self.transition_smooth_loss(),
        }


class PathConsistencyLoss(nn.Module):
    """Pairwise cosine loss for same-speaker, different-age batch pairs."""

    def __init__(self, ignore_age_index=-1, eps=1.0e-12):
        super().__init__()
        self.ignore_age_index = ignore_age_index
        self.eps = eps

    def valid_pair_indices(self, speakers, age_group):
        valid_age = age_group != self.ignore_age_index
        same_spk = speakers.view(-1, 1) == speakers.view(1, -1)
        diff_age = age_group.view(-1, 1) != age_group.view(1, -1)
        both_valid = valid_age.view(-1, 1) & valid_age.view(1, -1)
        upper = torch.triu(torch.ones_like(same_spk, dtype=torch.bool),
                           diagonal=1)
        return (same_spk & diff_age & both_valid & upper).nonzero(
            as_tuple=False)

    def forward(self, embeddings, speakers, age_group):
        pairs = self.valid_pair_indices(speakers, age_group)
        if pairs.numel() == 0:
            return _zero_like(embeddings)
        i, j = pairs[:, 0], pairs[:, 1]
        cosine = F.cosine_similarity(embeddings[i],
                                     embeddings[j],
                                     dim=-1,
                                     eps=self.eps)
        return (1.0 - cosine).mean()

    def valid_pair_count(self, embeddings, speakers, age_group):
        return embeddings.new_tensor(
            float(self.valid_pair_indices(speakers, age_group).size(0)))


def acsm_warmup_scale(epoch, ramp_epoch):
    ramp_epoch = int(ramp_epoch or 0)
    if ramp_epoch <= 0:
        return 1.0
    if epoch is None:
        return 1.0
    # WeSpeaker train.py uses 1-based epochs, while standalone tests may pass
    # 0-based epochs. In both cases the first epoch gets 1 / ramp_epoch.
    step = float(epoch + 1) if epoch <= 0 else float(epoch)
    return min(1.0, step / float(max(ramp_epoch, 1)))


def acsm_diagnostics(outputs, loss_total):
    raw_norm = F.normalize(outputs['raw_embedding'].detach(),
                           dim=-1,
                           eps=1.0e-12)
    can_norm = F.normalize(outputs['embedding'], dim=-1, eps=1.0e-12)
    raw_can_cos = (raw_norm * can_norm).sum(dim=-1)
    raw_can_l2 = (can_norm - raw_norm).norm(dim=-1)
    residual_norm = outputs['canonical_residual'].norm(dim=-1)
    return {
        'gate_mean': _stat(outputs['gate'].mean(), loss_total),
        'gate_std': _stat(outputs['gate'].std(unbiased=False), loss_total),
        'uncertainty_mean': _stat(outputs['uncertainty'].mean(), loss_total),
        'residual_norm': _stat(residual_norm.mean(), loss_total),
        'residual_norm_mean': _stat(residual_norm.mean(), loss_total),
        'cos_raw_can_mean': _stat(raw_can_cos.mean(), loss_total),
        'l2_raw_can_mean': _stat(raw_can_l2.mean(), loss_total),
    }
