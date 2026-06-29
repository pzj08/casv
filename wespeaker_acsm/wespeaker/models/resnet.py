# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
#               2024 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''ResNet in PyTorch.

Some modifications from the original architecture:
1. Smaller kernel size for the input layer
2. Smaller number of Channels
3. No max_pooling involved

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers
from wespeaker.models.acsm_modules import AgeFiLM2d
from wespeaker.models.acsm_modules import OrderedAgeCanonicalizer
from wespeaker.models.acsm_modules import PathConsistencyLoss
from wespeaker.models.acsm_modules import Stage2AgeObserver
from wespeaker.models.acsm_modules import acsm_diagnostics
from wespeaker.models.acsm_modules import acsm_warmup_scale
from wespeaker.models.acsm_modules import get_acsm_config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.pool = getattr(pooling_layers,
                            pooling_func)(in_dim=self.stats_dim *
                                          block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _get_frame_level_feat(self, x):
        # for inner class usage
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

    def get_frame_level_feat(self, x):
        # for outer interface
        out = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)

        return out  # (B, T, D)

    def forward(self, x):
        out = self._get_frame_level_feat(x)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


class ResNetACSM(ResNet):
    """Structural ACSM variant of WeSpeaker ResNet.

    The official ResNet stem, residual layers, pooling and segment layers keep
    their original names so a baseline ResNet checkpoint can partially
    initialize the shared encoder.
    """

    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=False,
                 acsm_args=None):
        super().__init__(block,
                         num_blocks,
                         m_channels=m_channels,
                         feat_dim=feat_dim,
                         embed_dim=embed_dim,
                         pooling_func=pooling_func,
                         two_emb_layer=two_emb_layer)
        self.acsm_config = get_acsm_config({
            'model': 'ResNet34_ACSM',
            'model_args': {
                'acsm_args': acsm_args or {}
            }
        })
        self.num_age_groups = int(self.acsm_config['num_age_groups'])
        self.ignore_age_index = int(self.acsm_config['ignore_age_index'])
        self.eps = float(self.acsm_config.get('eps', 1.0e-12))

        layer2_channels = m_channels * 2 * block.expansion
        layer3_channels = m_channels * 4 * block.expansion
        layer4_channels = m_channels * 8 * block.expansion
        self.age_observer = Stage2AgeObserver(
            layer2_channels,
            self.num_age_groups,
            age_emb_dim=int(self.acsm_config['age_emb_dim']),
            ignore_age_index=self.ignore_age_index,
            eps=self.eps)

        film_conf = self.acsm_config['film']
        film_enabled = bool(film_conf.get('enabled', True))
        film_stages = set(film_conf.get('stages', ['layer3', 'layer4']))
        film_scale = float(film_conf.get('film_scale', 0.05))
        self.age_film3 = AgeFiLM2d(layer3_channels,
                                   self.num_age_groups,
                                   film_scale=film_scale,
                                   enabled=film_enabled
                                   and 'layer3' in film_stages)
        self.age_film4 = AgeFiLM2d(layer4_channels,
                                   self.num_age_groups,
                                   film_scale=film_scale,
                                   enabled=film_enabled
                                   and 'layer4' in film_stages)

        canon_conf = self.acsm_config['canonicalizer']
        self.canonicalizer = OrderedAgeCanonicalizer(
            self.num_age_groups,
            embed_dim,
            int(self.acsm_config['reference_age_group']),
            canonical_scale=float(canon_conf.get('canonical_scale', 0.1)),
            learnable_canonical_scale=bool(
                canon_conf.get('learnable_canonical_scale', False)),
            gate_max=float(canon_conf.get('gate_max', 0.5)),
            gate_init_bias=float(canon_conf.get('gate_init_bias', -2.0)),
            transition_init_std=float(
                canon_conf.get('transition_init_std', 0.005)),
            enabled=bool(canon_conf.get('enabled', True)),
            eps=self.eps)
        self.path_loss = PathConsistencyLoss(self.ignore_age_index, self.eps)

    def _get_frame_level_feat(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        h2 = self.layer2(out)
        age_outputs = self.age_observer(h2)
        q_age = age_outputs['age_posterior']
        out = self.layer3(h2)
        out = self.age_film3(out, q_age)
        out = self.layer4(out)
        out = self.age_film4(out, q_age)
        return out, age_outputs

    def get_frame_level_feat(self, x):
        out, _ = self._get_frame_level_feat(x)
        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)
        return out

    def _observed_embedding(self, stats):
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            return self.seg_2(out)
        return embed_a

    def forward(self, x):
        frame_feat, age_outputs = self._get_frame_level_feat(x)
        stats = self.pool(frame_feat)
        raw_embedding = self._observed_embedding(stats)
        canon_outputs = self.canonicalizer(raw_embedding,
                                           age_outputs['age_posterior'])
        outputs = {
            'embedding': canon_outputs['embedding'],
            'raw_embedding': raw_embedding,
            'age_posterior': age_outputs['age_posterior'],
            'age_pred': age_outputs['age_pred'],
            'rank_logits': age_outputs['rank_logits'],
            'age_embedding': age_outputs['age_embedding'],
            'canonical_residual': canon_outputs['canonical_residual'],
            'gate': canon_outputs['gate'],
            'uncertainty': canon_outputs['uncertainty'],
            'path_norm': canon_outputs['path_norm'],
            'acsm_loss_inputs': {
                'transition_smooth_loss':
                canon_outputs['transition_smooth_loss']
            },
        }
        return outputs

    def compute_acsm_losses(self, outputs, speakers, age_groups, epoch=None):
        zero = outputs['embedding'].new_zeros(())
        loss_conf = self.acsm_config['losses']
        warm = acsm_warmup_scale(epoch, loss_conf.get('ramp_epoch', 0))
        losses = {
            'loss_age': zero,
            'loss_consistency': zero,
            'loss_smooth': zero,
            'loss_path': zero,
        }
        losses['path_valid_pair_count'] = self.path_loss.valid_pair_count(
            outputs['embedding'], speakers, age_groups)

        if float(loss_conf.get('lambda_age', 0.0)) > 0.0:
            losses['loss_age'] = self.age_observer.ordinal_loss(
                outputs['rank_logits'], age_groups)

        cons_conf = self.acsm_config.get('consistency', {})
        e_can = outputs['embedding']
        e_obs = outputs['raw_embedding'].detach()
        consistency_type = cons_conf.get('type', 'cosine')
        if consistency_type == 'cosine':
            e_can_n = F.normalize(e_can, p=2, dim=-1)
            e_obs_n = F.normalize(e_obs, p=2, dim=-1)
            consistency = (1.0 - F.cosine_similarity(e_can_n,
                                                      e_obs_n,
                                                      dim=-1)).clamp_min(0.0)
        elif consistency_type == 'raw_l2_sum':
            consistency = (e_can - e_obs).pow(2).sum(dim=-1)
        else:
            raise ValueError('unsupported ACSM consistency.type: {}'.format(
                consistency_type))
        if age_groups is not None and cons_conf.get('only_small_age_gap',
                                                    False):
            valid = age_groups != self.ignore_age_index
            if valid.any():
                consistency = consistency[valid]
        losses['loss_consistency'] = consistency.mean(
        ) if consistency.numel() > 0 else zero

        losses['loss_smooth'] = outputs['acsm_loss_inputs'][
            'transition_smooth_loss']
        if float(loss_conf.get('lambda_path', 0.0)) > 0.0:
            losses['loss_path'] = self.path_loss(e_can, speakers, age_groups)

        loss_total = (
            float(loss_conf.get('lambda_age', 0.0)) * losses['loss_age'] +
            float(loss_conf.get('lambda_consistency', 0.0)) *
            losses['loss_consistency'] +
            float(loss_conf.get('lambda_smooth', 0.0)) *
            losses['loss_smooth'] +
            float(loss_conf.get('lambda_path', 0.0)) * losses['loss_path'])
        loss_total = loss_total * float(warm)
        losses['loss_acsm_total'] = loss_total
        losses['weighted_consistency'] = (
            float(loss_conf.get('lambda_consistency', 0.0)) *
            losses['loss_consistency'] * float(warm))
        if self.acsm_config['diagnostics'].get('log_diagnostics', True):
            losses.update(acsm_diagnostics(outputs, loss_total))
        if self.acsm_config['diagnostics'].get('strict_finite_check', True):
            for name, value in losses.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError(
                        '{} is not finite in ACSM loss computation'.format(
                            name))
        return losses


class ResNetParamMatch(ResNet):
    """ResNet variant with non-age extra parameters for ACSM size control."""

    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=False,
                 param_match_args=None):
        super().__init__(block,
                         num_blocks,
                         m_channels=m_channels,
                         feat_dim=feat_dim,
                         embed_dim=embed_dim,
                         pooling_func=pooling_func,
                         two_emb_layer=two_emb_layer)
        conf = param_match_args or {}
        bottleneck_dim = int(conf.get('bottleneck_dim', 64))
        self.param_match_residual_scale = float(conf.get('residual_scale',
                                                        0.1))
        self.param_match_token = nn.Parameter(torch.zeros(embed_dim))
        self.param_match_norm = nn.LayerNorm(embed_dim)
        self.param_match_residual = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        self.param_match_gate = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 1),
        )

    def _embedding(self, stats):
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            raw = self.seg_2(out)
            first = embed_a
        else:
            raw = embed_a
            first = raw.new_zeros(())
        cond = self.param_match_norm(raw + self.param_match_token.view(1, -1))
        residual = self.param_match_residual(cond)
        gate = torch.sigmoid(self.param_match_gate(cond))
        embedding = raw + self.param_match_residual_scale * gate * residual
        return first, embedding

    def forward(self, x):
        out = self._get_frame_level_feat(x)
        stats = self.pool(out)
        return self._embedding(stats)


def ResNet18(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet34(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet34_ACSM(feat_dim,
                  embed_dim,
                  pooling_func='TSTP',
                  two_emb_layer=False,
                  acsm_args=None):
    return ResNetACSM(BasicBlock, [3, 4, 6, 3],
                      feat_dim=feat_dim,
                      embed_dim=embed_dim,
                      pooling_func=pooling_func,
                      two_emb_layer=two_emb_layer,
                      acsm_args=acsm_args)


def ACSM_ResNet34(feat_dim,
                  embed_dim,
                  pooling_func='TSTP',
                  two_emb_layer=False,
                  acsm_args=None):
    return ResNet34_ACSM(feat_dim,
                         embed_dim,
                         pooling_func=pooling_func,
                         two_emb_layer=two_emb_layer,
                         acsm_args=acsm_args)


def ResNet34_ParamMatch(feat_dim,
                        embed_dim,
                        pooling_func='TSTP',
                        two_emb_layer=False,
                        param_match_args=None):
    return ResNetParamMatch(BasicBlock, [3, 4, 6, 3],
                            feat_dim=feat_dim,
                            embed_dim=embed_dim,
                            pooling_func=pooling_func,
                            two_emb_layer=two_emb_layer,
                            param_match_args=param_match_args)


def ResNet50(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet101(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet152(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return ResNet(Bottleneck, [3, 8, 36, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet221(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return ResNet(Bottleneck, [6, 16, 48, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet293(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=False):
    return ResNet(Bottleneck, [10, 20, 64, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)
    model = ResNet34(feat_dim=80, embed_dim=256, two_emb_layer=False)
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))

    # from torchinfo import summary
    # summary(model, (16, 100, 80))
