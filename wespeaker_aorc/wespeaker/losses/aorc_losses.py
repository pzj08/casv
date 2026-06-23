# Copyright (c) 2026
#
# Licensed under the Apache License, Version 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_like(x):
    return x.new_zeros(())


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

    def forward(self, z_age, prototypes, speakers, age_group):
        valid_age = age_group != self.ignore_index
        same_spk = speakers.view(-1, 1) == speakers.view(1, -1)
        diff_age = age_group.view(-1, 1) != age_group.view(1, -1)
        both_valid = valid_age.view(-1, 1) & valid_age.view(1, -1)
        younger_to_older = age_group.view(-1, 1) < age_group.view(1, -1)
        mask = same_spk & diff_age & both_valid & younger_to_older
        pairs = mask.nonzero(as_tuple=False)
        if pairs.numel() == 0:
            return _zero_like(z_age)
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
        return (omega * (1.0 - cosine)).mean()


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
