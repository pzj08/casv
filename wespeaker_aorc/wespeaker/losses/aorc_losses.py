# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_like(x):
    return x.new_zeros(())


def _stat(value, ref):
    if torch.is_tensor(value):
        return value.detach()
    return ref.new_tensor(float(value)).detach()


def _finite_flag(value, ref):
    if not torch.is_tensor(value):
        return _stat(float(torch.isfinite(ref.new_tensor(value))), ref)
    return _stat(torch.isfinite(value).all().to(ref.dtype), ref)


class OrdinalAgeLoss(nn.Module):
    """CORAL-style ordinal regression loss for ordered age groups."""

    def __init__(self, num_age_groups, ignore_index=-1):
        super().__init__()
        self.num_age_groups = num_age_groups
        self.ignore_index = ignore_index
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, rank_logits, age_group):
        valid = age_group != self.ignore_index
        if valid.sum() == 0:
            return _zero_like(rank_logits)
        logits = rank_logits[valid]
        labels = age_group[valid].long()
        thresholds = torch.arange(self.num_age_groups - 1,
                                  device=logits.device).view(1, -1)
        targets = (labels.view(-1, 1) > thresholds).to(logits.dtype)
        return self.loss(logits, targets)


class OrdinalPrototypeLoss(nn.Module):
    """Age-distance weighted prototype classification loss."""

    def __init__(self,
                 num_age_groups,
                 tau=0.1,
                 lambda_proto_dist=1.0,
                 ignore_index=-1,
                 eps=1e-12):
        super().__init__()
        self.num_age_groups = num_age_groups
        self.tau = tau
        self.lambda_proto_dist = lambda_proto_dist
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, z_age, prototypes, age_group):
        valid = age_group != self.ignore_index
        if valid.sum() == 0:
            return _zero_like(z_age)
        z_age = F.normalize(z_age[valid], dim=-1, eps=self.eps)
        labels = age_group[valid].long()
        prototypes = F.normalize(prototypes, dim=-1, eps=self.eps)
        logits = torch.matmul(z_age, prototypes.t()) / self.tau

        groups = torch.arange(self.num_age_groups, device=z_age.device)
        denom = max(self.num_age_groups - 1, 1)
        dist = (labels.view(-1, 1) - groups.view(1, -1)).abs().float() / denom
        alpha = 1.0 + self.lambda_proto_dist * dist
        weighted_lse = torch.logsumexp(
            logits + alpha.clamp_min(self.eps).log(), dim=1)
        positive = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        return (weighted_lse - positive).mean()


class SoftProxyMatchingLoss(nn.Module):
    """Proxy matching with age-distance weights on negative proxies only."""

    def __init__(self,
                 num_age_groups,
                 tau=0.1,
                 lambda_proto_dist=1.0,
                 ignore_index=-1,
                 eps=1e-12,
                 weight_type="sigmoid",
                 include_positive_in_denominator=True):
        super().__init__()
        self.num_age_groups = num_age_groups
        self.tau = tau
        self.lambda_proto_dist = lambda_proto_dist
        self.ignore_index = ignore_index
        self.eps = eps
        self.weight_type = weight_type
        self.include_positive_in_denominator = include_positive_in_denominator

    def _empty_stats(self, ref):
        return {
            'proxy_valid_count': _stat(0.0, ref),
            'proxy_weight_min': _stat(0.0, ref),
            'proxy_weight_mean': _stat(0.0, ref),
            'proxy_weight_max': _stat(0.0, ref),
            'proxy_loss_is_finite': _stat(1.0, ref),
        }

    def _weights(self, labels, device, dtype):
        groups = torch.arange(self.num_age_groups, device=device)
        denom = max(self.num_age_groups - 1, 1)
        dist = (labels.view(-1, 1) - groups.view(1, -1)).abs().to(dtype)
        dist = dist / float(denom)
        if self.weight_type == "none":
            weights = torch.ones_like(dist)
        elif self.weight_type == "linear":
            weights = 1.0 + self.lambda_proto_dist * dist
        elif self.weight_type == "sigmoid":
            weights = 1.0 + self.lambda_proto_dist * torch.sigmoid(dist)
        else:
            raise ValueError('unsupported proxy weight_type: {}'.format(
                self.weight_type))
        positive = groups.view(1, -1) == labels.view(-1, 1)
        weights = torch.where(positive, torch.ones_like(weights), weights)
        return weights.clamp_min(self.eps)

    def forward(self, z_age, prototypes, age_group, return_stats=False):
        valid = ((age_group != self.ignore_index) & (age_group >= 0)
                 & (age_group < self.num_age_groups))
        if valid.sum() == 0:
            loss = _zero_like(z_age)
            stats = self._empty_stats(z_age)
            return (loss, stats) if return_stats else loss

        z_valid = F.normalize(z_age[valid], dim=-1, eps=self.eps)
        labels = age_group[valid].long()
        proto = F.normalize(prototypes, dim=-1, eps=self.eps)
        logits = torch.matmul(z_valid, proto.t()) / self.tau
        logits_f = logits.float()

        weights = self._weights(labels, logits.device, logits_f.dtype)
        groups = torch.arange(self.num_age_groups, device=logits.device)
        positive_mask = groups.view(1, -1) == labels.view(-1, 1)
        positive = logits_f.gather(1, labels.view(-1, 1)).squeeze(1)

        if self.include_positive_in_denominator:
            weighted_logits = logits_f + weights.log()
            denom = torch.logsumexp(weighted_logits, dim=1)
            loss = (denom - positive).mean()
        else:
            negative_logits = logits_f + weights.log()
            negative_logits = negative_logits.masked_fill(
                positive_mask, float('-inf'))
            has_negative = (~positive_mask).any(dim=1)
            if has_negative.any():
                neg_lse = torch.logsumexp(negative_logits[has_negative],
                                          dim=1)
                loss = F.softplus(neg_lse - positive[has_negative]).mean()
            else:
                loss = _zero_like(z_age).float()
        loss = loss.to(z_age.dtype)
        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        detached_weights = weights.detach()
        stats = {
            'proxy_valid_count': _stat(valid.sum().to(z_age.dtype), z_age),
            'proxy_weight_min': _stat(detached_weights.min(), z_age),
            'proxy_weight_mean': _stat(detached_weights.mean(), z_age),
            'proxy_weight_max': _stat(detached_weights.max(), z_age),
            'proxy_loss_is_finite': _finite_flag(loss, z_age),
        }
        return (loss, stats) if return_stats else loss


class SpeakerConditionedDirectionLoss(nn.Module):
    """Direction consistency for same-speaker cross-age pairs."""

    def __init__(self,
                 num_age_groups,
                 beta_gap=1.0,
                 ignore_index=-1,
                 max_pairs=None,
                 eps=1e-12):
        super().__init__()
        self.num_age_groups = num_age_groups
        self.beta_gap = beta_gap
        self.ignore_index = ignore_index
        self.max_pairs = max_pairs
        self.eps = eps

    def _empty_stats(self, ref):
        return {
            'dir_active': _stat(0.0, ref),
            'dir_num_pairs': _stat(0.0, ref),
            'dir_num_pairs_before_subsample': _stat(0.0, ref),
            'dir_num_pairs_after_subsample': _stat(0.0, ref),
            'dir_mean_gap': _stat(0.0, ref),
            'dir_cosine_mean': _stat(0.0, ref),
            'dir_omega_min': _stat(0.0, ref),
            'dir_omega_mean': _stat(0.0, ref),
            'dir_omega_max': _stat(0.0, ref),
            'dir_loss_is_finite': _stat(1.0, ref),
        }

    def forward(self,
                z_age,
                prototypes,
                speakers,
                age_group,
                return_stats=False):
        valid_age = age_group != self.ignore_index
        same_spk = speakers.view(-1, 1) == speakers.view(1, -1)
        diff_age = age_group.view(-1, 1) != age_group.view(1, -1)
        both_valid = valid_age.view(-1, 1) & valid_age.view(1, -1)
        younger_to_older = age_group.view(-1, 1) < age_group.view(1, -1)
        mask = same_spk & diff_age & both_valid & younger_to_older
        pairs = mask.nonzero(as_tuple=False)
        num_before = pairs.size(0)
        if pairs.numel() == 0:
            loss = _zero_like(z_age)
            stats = self._empty_stats(z_age)
            return (loss, stats) if return_stats else loss
        if self.max_pairs is not None and pairs.size(0) > self.max_pairs:
            perm = torch.randperm(pairs.size(0), device=pairs.device)
            pairs = pairs[perm[:self.max_pairs]]

        i, j = pairs[:, 0], pairs[:, 1]
        proto = F.normalize(prototypes, dim=-1, eps=self.eps)
        v_age = F.normalize(z_age[j] - z_age[i], dim=-1, eps=self.eps)
        v_proto = F.normalize(proto[age_group[j].long()] -
                              proto[age_group[i].long()],
                              dim=-1,
                              eps=self.eps)
        cosine = (v_age * v_proto).sum(dim=-1)
        denom = max(self.num_age_groups - 1, 1)
        gap = (age_group[j] - age_group[i]).abs().float() / denom
        omega = 1.0 + self.beta_gap * gap
        loss = (omega * (1.0 - cosine)).mean()
        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        stats = {
            'dir_active': _stat(1.0, z_age),
            'dir_num_pairs': _stat(float(pairs.size(0)), z_age),
            'dir_num_pairs_before_subsample': _stat(float(num_before), z_age),
            'dir_num_pairs_after_subsample': _stat(float(pairs.size(0)),
                                                   z_age),
            'dir_mean_gap': _stat(gap.mean(), z_age),
            'dir_cosine_mean': _stat(cosine.mean(), z_age),
            'dir_omega_min': _stat(omega.min(), z_age),
            'dir_omega_mean': _stat(omega.mean(), z_age),
            'dir_omega_max': _stat(omega.max(), z_age),
            'dir_loss_is_finite': _finite_flag(loss, z_age),
        }
        return (loss, stats) if return_stats else loss


class CrossAgeAggregationLoss(nn.Module):
    """Age-gap-aware supervised contrastive loss over speaker embeddings."""

    def __init__(self,
                 num_age_groups,
                 tau=0.07,
                 gamma_caa=1.0,
                 ignore_index=-1,
                 normalize_weights=True,
                 eps=1e-12):
        super().__init__()
        self.num_age_groups = num_age_groups
        self.tau = tau
        self.gamma_caa = gamma_caa
        self.ignore_index = ignore_index
        self.normalize_weights = normalize_weights
        self.eps = eps

    def forward(self, z_spk, speakers, age_group):
        z_spk = F.normalize(z_spk, dim=-1, eps=self.eps)
        batch = z_spk.size(0)
        sim = torch.matmul(z_spk, z_spk.t()) / self.tau
        eye = torch.eye(batch, dtype=torch.bool, device=z_spk.device)
        sim = sim.masked_fill(eye, float('-inf'))
        log_den = torch.logsumexp(sim, dim=1)

        positive = (speakers.view(-1, 1) == speakers.view(1, -1)) & ~eye
        if positive.sum() == 0:
            return _zero_like(z_spk)

        valid_age = age_group != self.ignore_index
        both_valid = valid_age.view(-1, 1) & valid_age.view(1, -1)
        denom = max(self.num_age_groups - 1, 1)
        gap = (age_group.view(-1, 1) - age_group.view(1, -1)).abs().float()
        eta = torch.ones_like(sim)
        eta = torch.where(both_valid, 1.0 + self.gamma_caa * gap / denom, eta)
        eta = eta.masked_select(positive)
        if self.normalize_weights:
            eta = eta / eta.mean().detach().clamp_min(self.eps)

        log_prob = sim - log_den.view(-1, 1)
        per_pair = -eta * log_prob.masked_select(positive)
        anchor_counts = positive.sum(dim=1)
        valid_anchor = anchor_counts > 0
        pair_anchor = positive.nonzero(as_tuple=False)[:, 0]
        anchor_loss = z_spk.new_zeros(batch)
        anchor_loss.scatter_add_(0, pair_anchor, per_pair)
        anchor_loss[valid_anchor] = (
            anchor_loss[valid_anchor] / anchor_counts[valid_anchor].float())
        return anchor_loss[valid_anchor].mean()


class CrossAgeAggregationLossV2(nn.Module):
    """Safer CAA variant with explicit cross-age positive modes."""

    _MODE_IDS = {
        "disabled": 0,
        "all_same_speaker": 1,
        "cross_age_only": 2,
        "large_gap_only": 3,
    }

    def __init__(self,
                 num_age_groups,
                 tau=0.10,
                 gamma_caa=0.5,
                 ignore_index=-1,
                 normalize_weights=True,
                 mode="cross_age_only",
                 min_gap=1,
                 eta_max=2.0,
                 eps=1e-12):
        super().__init__()
        if mode not in self._MODE_IDS:
            raise ValueError('unsupported CAA mode: {}'.format(mode))
        self.num_age_groups = num_age_groups
        self.tau = tau
        self.gamma_caa = gamma_caa
        self.ignore_index = ignore_index
        self.normalize_weights = normalize_weights
        self.mode = mode
        self.min_gap = min_gap
        self.eta_max = eta_max
        self.eps = eps

    def _base_stats(self, ref, age_gap=None, eta=None, positive=None):
        if age_gap is None:
            mean_gap = _stat(0.0, ref)
        else:
            mean_gap = _stat(age_gap.mean(), ref)
        if eta is None or eta.numel() == 0:
            eta_min = eta_mean = eta_max = _stat(0.0, ref)
        else:
            eta_min = _stat(eta.min(), ref)
            eta_mean = _stat(eta.mean(), ref)
            eta_max = _stat(eta.max(), ref)
        return {
            'caa_active': _stat(0.0, ref),
            'caa_mode_id': _stat(float(self._MODE_IDS[self.mode]), ref),
            'caa_num_positive_pairs': _stat(
                float(positive.sum().item()) if positive is not None else 0.0,
                ref),
            'caa_num_cross_age_positive_pairs': _stat(0.0, ref),
            'caa_num_large_gap_positive_pairs': _stat(0.0, ref),
            'caa_valid_anchor_count': _stat(0.0, ref),
            'caa_mean_age_gap': mean_gap,
            'caa_eta_min': eta_min,
            'caa_eta_mean': eta_mean,
            'caa_eta_max': eta_max,
            'caa_loss_is_finite': _stat(1.0, ref),
        }

    def forward(self, z_spk, speakers, age_group, return_stats=False):
        if self.mode == "disabled":
            loss = _zero_like(z_spk)
            stats = self._base_stats(z_spk)
            return (loss, stats) if return_stats else loss

        z_spk = F.normalize(z_spk, dim=-1, eps=self.eps)
        batch = z_spk.size(0)
        if batch <= 1:
            loss = _zero_like(z_spk)
            stats = self._base_stats(z_spk)
            return (loss, stats) if return_stats else loss

        sim = torch.matmul(z_spk, z_spk.t()).float() / self.tau
        eye = torch.eye(batch, dtype=torch.bool, device=z_spk.device)
        sim = sim.masked_fill(eye, float('-inf'))
        log_den = torch.logsumexp(sim, dim=1)

        same_spk = (speakers.view(-1, 1) == speakers.view(1, -1)) & ~eye
        valid_age = age_group != self.ignore_index
        both_valid = valid_age.view(-1, 1) & valid_age.view(1, -1)
        gap_raw = (age_group.view(-1, 1) - age_group.view(1, -1)).abs()
        cross_age = same_spk & both_valid & (gap_raw >= 1)
        large_gap = same_spk & both_valid & (gap_raw >= self.min_gap)

        if self.mode == "all_same_speaker":
            positive = same_spk
        elif self.mode == "cross_age_only":
            positive = same_spk & both_valid & (gap_raw >= self.min_gap)
        else:
            positive = large_gap

        denom = max(self.num_age_groups - 1, 1)
        gap = gap_raw.float() / float(denom)
        eta_full = 1.0 + self.gamma_caa * gap
        eta_full = eta_full.clamp_min(self.eps).clamp_max(self.eta_max)
        eta = eta_full.masked_select(positive)
        if eta.numel() > 0 and self.normalize_weights:
            eta = eta / eta.mean().detach().clamp_min(self.eps)
            eta = eta.clamp_min(self.eps).clamp_max(self.eta_max)

        stats = self._base_stats(z_spk,
                                 age_gap=gap.masked_select(positive)
                                 if positive.any() else None,
                                 eta=eta,
                                 positive=positive)
        stats['caa_num_cross_age_positive_pairs'] = _stat(
            float(cross_age.sum().item()), z_spk)
        stats['caa_num_large_gap_positive_pairs'] = _stat(
            float(large_gap.sum().item()), z_spk)

        if positive.sum() == 0:
            loss = _zero_like(z_spk)
            return (loss, stats) if return_stats else loss

        anchor_counts = positive.sum(dim=1)
        valid_anchor = anchor_counts > 0
        log_prob = sim - log_den.view(-1, 1)
        per_pair = -eta * log_prob.masked_select(positive)
        pair_anchor = positive.nonzero(as_tuple=False)[:, 0]
        anchor_loss = z_spk.new_zeros(batch, dtype=per_pair.dtype)
        anchor_loss.scatter_add_(0, pair_anchor, per_pair)
        anchor_loss[valid_anchor] = (
            anchor_loss[valid_anchor] / anchor_counts[valid_anchor].float())
        loss = anchor_loss[valid_anchor].mean().to(z_spk.dtype)
        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        stats['caa_active'] = _stat(1.0, z_spk)
        stats['caa_valid_anchor_count'] = _stat(
            float(valid_anchor.sum().item()), z_spk)
        stats['caa_loss_is_finite'] = _finite_flag(loss, z_spk)
        return (loss, stats) if return_stats else loss
