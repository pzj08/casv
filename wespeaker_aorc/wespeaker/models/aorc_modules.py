# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import warnings

from wespeaker.losses.aorc_losses import CrossAgeAggregationLoss
from wespeaker.losses.aorc_losses import CrossAgeAggregationLossV2
from wespeaker.losses.aorc_losses import OrdinalAgeLoss
from wespeaker.losses.aorc_losses import OrdinalPrototypeLoss
from wespeaker.losses.aorc_losses import SoftProxyMatchingLoss
from wespeaker.losses.aorc_losses import SpeakerConditionedDirectionLoss


AORC_DEFAULTS = {
    'num_age_groups': 7,
    'age_bins': None,
    'age_label_file': None,
    'age_label_type': 'group',
    'enable_oam': False,
    'enable_orc': False,
    'enable_caa': False,
    'age_mode': 'ordinal',
    'age_emb_dim': None,
    'lambda_oam': 0.1,
    'alpha_proxy': 0.1,
    'beta_dir': 0.05,
    'lambda_caa': 0.05,
    'lambda_smooth': 1.0e-3,
    'tau_proxy': 0.1,
    'tau_caa': 0.07,
    'lambda_proto_dist': 1.0,
    'proxy_loss_type': 'soft',
    'proxy_weight_type': 'sigmoid',
    'proxy_include_positive_in_denominator': True,
    'beta_gap': 1.0,
    'gamma_caa': 1.0,
    'caa_version': 'v2',
    'caa_mode': 'cross_age_only',
    'caa_min_gap': 1,
    'caa_eta_max': 2.0,
    'caa_warmup_epoch': 0,
    'caa_ramp_epoch': 0,
    'log_aorc_diagnostics': True,
    'aorc_strict_finite_check': True,
    'residual_scale': 1.0,
    'learnable_residual_scale': False,
    'initial_residual_scale': None,
    'detach_age_prob_for_residual': True,
    'ignore_age_index': -1,
    'max_dir_pairs': None,
    'age_head_hidden_dim': None,
    'eps': 1e-12,
}


def get_aorc_config(configs):
    conf = dict(AORC_DEFAULTS)
    conf.update(configs.get('aorc_args', {}))
    for key in AORC_DEFAULTS:
        if key in configs:
            conf[key] = configs[key]
    if conf['age_bins'] is not None:
        conf['num_age_groups'] = len(conf['age_bins']) + 1
    return conf


def aorc_is_enabled(configs):
    conf = get_aorc_config(configs)
    return bool(conf['enable_oam'] or conf['enable_orc'] or conf['enable_caa'])


def _distributed_ready():
    return (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1)


def _distributed_loss_scale():
    return float(dist.get_world_size()) if _distributed_ready() else 1.0


def _scale_tensor_gradient(tensor, scale):
    if scale == 1.0:
        return tensor
    return tensor.detach() + (tensor - tensor.detach()) * scale


def _all_gather_no_grad(tensor):
    if not _distributed_ready():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def _all_gather_with_local_grad(tensor):
    if not _distributed_ready():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


class OrdinalAgeHead(nn.Module):

    def __init__(self,
                 input_dim,
                 num_age_groups,
                 age_emb_dim=None,
                 age_mode='ordinal',
                 hidden_dim=None,
                 eps=1e-12):
        super().__init__()
        self.input_dim = input_dim
        self.num_age_groups = num_age_groups
        self.age_emb_dim = age_emb_dim or input_dim
        self.age_mode = age_mode
        self.eps = eps

        if hidden_dim is None:
            self.proj = nn.Linear(input_dim, self.age_emb_dim)
        else:
            self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, self.age_emb_dim))
        self.score = nn.Linear(self.age_emb_dim, 1)
        self.raw_delta = nn.Parameter(torch.zeros(num_age_groups - 1))
        if self.age_mode == 'ce':
            self.age_classifier = nn.Linear(self.age_emb_dim, num_age_groups)
        else:
            self.age_classifier = None
        self.prototypes = nn.Parameter(
            torch.empty(num_age_groups, self.age_emb_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.01)

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

    def forward(self, h):
        z_age = F.normalize(self.proj(h), dim=-1, eps=self.eps)
        if self.age_mode == 'ce':
            age_logits = self.age_classifier(z_age)
            q_age = F.softmax(age_logits, dim=-1)
            rank_logits = h.new_zeros(h.size(0), self.num_age_groups - 1)
        else:
            age_logits = h.new_zeros(h.size(0), self.num_age_groups)
            score = self.score(z_age)
            rank_logits = score - self._thresholds().view(1, -1)
            q_age = self._ordinal_distribution(rank_logits)
        return {
            'age_embedding': z_age,
            'age_distribution': q_age,
            'age_logits': age_logits,
            'rank_logits': rank_logits,
            'age_prototypes': self.prototypes,
        }


class AgeResidualCompensation(nn.Module):

    def __init__(self,
                 num_age_groups,
                 embed_dim,
                 residual_scale=1.0,
                 learnable_residual_scale=False,
                 initial_residual_scale=None,
                 eps=1e-12):
        super().__init__()
        self.eps = eps
        self.residual_basis = nn.Parameter(torch.zeros(num_age_groups,
                                                       embed_dim))
        nn.init.normal_(self.residual_basis, mean=0.0, std=0.01)
        init_scale = residual_scale if initial_residual_scale is None else (
            initial_residual_scale)
        if learnable_residual_scale:
            self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer('residual_scale',
                                 torch.tensor(float(residual_scale)))

    def smoothness_loss(self):
        if self.residual_basis.size(0) < 2:
            return self.residual_basis.new_zeros(())
        diffs = self.residual_basis[1:] - self.residual_basis[:-1]
        return diffs.pow(2).sum(dim=-1).mean()

    def forward(self, raw_embedding, q_age):
        residual = torch.matmul(q_age, self.residual_basis)
        embedding = F.normalize(raw_embedding - self.residual_scale * residual,
                                dim=-1,
                                eps=self.eps)
        return embedding, residual


class AORCWrapper(nn.Module):

    def __init__(self, encoder, embed_dim, config):
        super().__init__()
        self.encoder = encoder
        self.config = dict(config)
        self.enable_oam = bool(config['enable_oam'])
        self.enable_orc = bool(config['enable_orc'])
        self.enable_caa = bool(config['enable_caa'])
        self.age_mode = config['age_mode']
        self.ignore_age_index = config['ignore_age_index']
        self.eps = config['eps']
        self.num_age_groups = config['num_age_groups']

        if self.enable_orc and not self.enable_oam:
            raise ValueError('enable_orc=true requires enable_oam=true')
        if self.enable_caa and not self.enable_oam:
            raise ValueError('enable_caa=true requires enable_oam=true')
        if (self.enable_orc and config['detach_age_prob_for_residual']
                and float(config['lambda_oam']) == 0.0):
            warnings.warn(
                'enable_orc=true with detach_age_prob_for_residual=true and '
                'lambda_oam=0 leaves age_distribution weakly supervised; '
                'residual compensation may be ineffective.',
                RuntimeWarning)

        self.age_head = OrdinalAgeHead(
            input_dim=embed_dim,
            num_age_groups=self.num_age_groups,
            age_emb_dim=config['age_emb_dim'] or embed_dim,
            age_mode=self.age_mode,
            hidden_dim=config['age_head_hidden_dim'],
            eps=self.eps)
        self.residual = AgeResidualCompensation(
            num_age_groups=self.num_age_groups,
            embed_dim=embed_dim,
            residual_scale=config['residual_scale'],
            learnable_residual_scale=config['learnable_residual_scale'],
            initial_residual_scale=config['initial_residual_scale'],
            eps=self.eps)
        self.loss_ord = OrdinalAgeLoss(self.num_age_groups,
                                       self.ignore_age_index)
        proxy_loss_type = config['proxy_loss_type']
        if proxy_loss_type == 'legacy':
            self.loss_proxy = OrdinalPrototypeLoss(
                self.num_age_groups,
                tau=config['tau_proxy'],
                lambda_proto_dist=config['lambda_proto_dist'],
                ignore_index=self.ignore_age_index,
                eps=self.eps)
        elif proxy_loss_type == 'soft':
            self.loss_proxy = SoftProxyMatchingLoss(
                self.num_age_groups,
                tau=config['tau_proxy'],
                lambda_proto_dist=config['lambda_proto_dist'],
                ignore_index=self.ignore_age_index,
                eps=self.eps,
                weight_type=config['proxy_weight_type'],
                include_positive_in_denominator=config[
                    'proxy_include_positive_in_denominator'])
        else:
            raise ValueError('unsupported proxy_loss_type: {}'.format(
                proxy_loss_type))
        self.loss_dir = SpeakerConditionedDirectionLoss(
            self.num_age_groups,
            beta_gap=config['beta_gap'],
            ignore_index=self.ignore_age_index,
            max_pairs=config['max_dir_pairs'],
            eps=self.eps)
        if config['caa_version'] == 'legacy':
            self.loss_caa = CrossAgeAggregationLoss(
                self.num_age_groups,
                tau=config['tau_caa'],
                gamma_caa=config['gamma_caa'],
                ignore_index=self.ignore_age_index,
                eps=self.eps)
        elif config['caa_version'] == 'v2':
            self.loss_caa = CrossAgeAggregationLossV2(
                self.num_age_groups,
                tau=config['tau_caa'],
                gamma_caa=config['gamma_caa'],
                ignore_index=self.ignore_age_index,
                mode=config['caa_mode'],
                min_gap=config['caa_min_gap'],
                eta_max=config['caa_eta_max'],
                eps=self.eps)
        else:
            raise ValueError('unsupported caa_version: {}'.format(
                config['caa_version']))

    def _extract_embedding(self, outputs):
        return outputs[-1] if isinstance(outputs, tuple) else outputs

    def forward(self, x):
        base_outputs = self.encoder(x)
        raw_embedding = self._extract_embedding(base_outputs)
        if not (self.enable_oam or self.enable_orc or self.enable_caa):
            return base_outputs

        age_outputs = self.age_head(raw_embedding)
        q_age = age_outputs['age_distribution']
        if self.enable_orc:
            q_for_residual = q_age.detach() if self.config[
                'detach_age_prob_for_residual'] else q_age
            embedding, residual = self.residual(raw_embedding, q_for_residual)
        else:
            embedding = raw_embedding
            residual = raw_embedding.new_zeros(raw_embedding.shape)

        return {
            'embedding': embedding,
            'raw_embedding': raw_embedding,
            'residual': residual,
            **age_outputs,
        }

    def _stat(self, name, value, ref):
        if not torch.is_tensor(value):
            value = ref.new_tensor(float(value))
        return ('stat_' + name, value.detach())

    def caa_lambda_for_epoch(self, epoch=None):
        if not self.enable_caa:
            return 0.0
        base = float(self.config['lambda_caa'])
        if epoch is None:
            return base
        warmup = int(self.config['caa_warmup_epoch'] or 0)
        ramp = int(self.config['caa_ramp_epoch'] or 0)
        if epoch < warmup:
            return 0.0
        if ramp > 0 and epoch < warmup + ramp:
            ramp_step = max(epoch - warmup, 0)
            return base * float(ramp_step) / float(ramp)
        return base

    def compute_aorc_losses(self, outputs, speakers, age_group, epoch=None):
        zero = outputs['embedding'].new_zeros(())
        global_speakers = _all_gather_no_grad(speakers)
        global_age_group = _all_gather_no_grad(age_group)
        global_age_embedding = _all_gather_with_local_grad(
            outputs['age_embedding'])
        global_embedding = _all_gather_with_local_grad(outputs['embedding'])
        losses = {
            'loss_oam': zero,
            'loss_ord': zero,
            'loss_proxy': zero,
            'loss_dir': zero,
            'loss_caa': zero,
            'loss_smooth': zero,
        }
        diagnostics = {}
        if self.enable_oam:
            if self.age_mode == 'ce':
                valid = age_group != self.ignore_age_index
                if valid.sum() > 0:
                    losses['loss_ord'] = F.cross_entropy(
                        outputs['age_logits'][valid], age_group[valid].long())
                diagnostics['proxy_valid_count'] = valid.sum().to(zero.dtype)
            else:
                losses['loss_ord'] = self.loss_ord(outputs['rank_logits'],
                                                   age_group)
                proxy_result = self.loss_proxy(
                    outputs['age_embedding'],
                    outputs['age_prototypes'],
                    age_group,
                    return_stats=True) if isinstance(
                        self.loss_proxy, SoftProxyMatchingLoss) else (
                            self.loss_proxy(outputs['age_embedding'],
                                            outputs['age_prototypes'],
                                            age_group), {})
                losses['loss_proxy'], proxy_stats = proxy_result
                diagnostics.update(proxy_stats)
                dir_age_embedding = _scale_tensor_gradient(
                    global_age_embedding, _distributed_loss_scale())
                losses['loss_dir'], dir_stats = self.loss_dir(
                    dir_age_embedding,
                    outputs['age_prototypes'],
                    global_speakers,
                    global_age_group,
                    return_stats=True)
                diagnostics.update(dir_stats)
            losses['loss_oam'] = (
                losses['loss_ord'] +
                self.config['alpha_proxy'] * losses['loss_proxy'] +
                self.config['beta_dir'] * losses['loss_dir'])

        if self.enable_orc:
            losses['loss_smooth'] = self.residual.smoothness_loss()
        if self.enable_caa:
            if isinstance(self.loss_caa, CrossAgeAggregationLossV2):
                losses['loss_caa'], caa_stats = self.loss_caa(
                    global_embedding,
                    global_speakers,
                    global_age_group,
                    return_stats=True)
                diagnostics.update(caa_stats)
            else:
                losses['loss_caa'] = self.loss_caa(global_embedding,
                                                   global_speakers,
                                                   global_age_group)
            losses['loss_caa'] = losses['loss_caa'] * _distributed_loss_scale()
        diagnostics['caa_lambda_eff'] = zero.new_tensor(
            self.caa_lambda_for_epoch(epoch))
        if self.config['aorc_strict_finite_check']:
            for name, value in losses.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(
                        '{} is not finite in AORC loss computation'.format(
                            name))
        if self.config['log_aorc_diagnostics']:
            for name, value in diagnostics.items():
                stat_name, stat_value = self._stat(name, value, zero)
                losses[stat_name] = stat_value
        return losses

    def residual_scale_value(self):
        return self.residual.residual_scale.detach()
