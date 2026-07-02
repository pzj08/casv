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
from wespeaker.models.cvac_waveform_modules import (
    DifferentiableLogMelFrontend, EmbeddingAgeHead, WaveformCVACLoss,
    WaveformCounterfactualAgingGenerator)


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
                 two_emb_layer=False,
                 cvac_args=None):
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

        self.cvac_args = cvac_args or {}
        self.cvac_enabled = bool(self.cvac_args.get('enabled', False))
        if self.cvac_enabled:
            num_age_groups = self.cvac_args.get('num_age_groups', 7)
            self.cvac_age_head = EmbeddingAgeHead(
                embed_dim,
                num_age_groups=num_age_groups,
                age_hidden_dim=self.cvac_args.get('age_hidden_dim', 256),
                age_dropout=self.cvac_args.get('age_dropout', 0.1),
                ignore_age_index=self.cvac_args.get('ignore_age_index', -1))
            self.cvac_generator = WaveformCounterfactualAgingGenerator(
                embedding_dim=embed_dim,
                num_age_groups=num_age_groups,
                condition_dim=self.cvac_args.get('condition_dim', 256),
                hidden_channels=self.cvac_args.get('hidden_channels', 128),
                num_layers=self.cvac_args.get('num_layers', 6),
                kernel_size=self.cvac_args.get('kernel_size', 7),
                dilations=self.cvac_args.get('dilations', None),
                residual_scale=self.cvac_args.get('residual_scale', 0.03),
                gate_max=self.cvac_args.get('gate_max', 0.15),
                gate_init_bias=self.cvac_args.get('gate_init_bias', -3.0))
            self.cvac_loss = WaveformCVACLoss(**self.cvac_args)
            self.cvac_logmel = DifferentiableLogMelFrontend(
                sample_rate=self.cvac_args.get('sample_rate', 16000),
                num_mel_bins=self.cvac_args.get('mel_bins', feat_dim),
                n_fft=self.cvac_args.get('mel_fft_size', 512),
                hop_length=self.cvac_args.get('mel_hop_size', 160),
                win_length=self.cvac_args.get('mel_win_length', 400),
                eps=self.cvac_args.get('eps', 1e-12))
        else:
            self.cvac_age_head = None
            self.cvac_generator = None
            self.cvac_loss = None
            self.cvac_logmel = None

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

    def _forward_features(self, x):
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

    def forward(self, x):
        return self._forward_features(x)

    def forward_waveform_for_cvac(self, waveform):
        """CVAC-only differentiable waveform -> logmel -> speaker embedding."""
        if self.cvac_logmel is None:
            raise RuntimeError('CVAC waveform frontend is not initialized')
        feats = self.cvac_logmel(waveform)
        return self._forward_features(feats)[-1]


def ResNet18(feat_dim,
             embed_dim,
             pooling_func='TSTP',
             two_emb_layer=False,
             cvac_args=None):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer,
                  cvac_args=cvac_args)


def ResNet34(feat_dim,
             embed_dim,
             pooling_func='TSTP',
             two_emb_layer=False,
             cvac_args=None):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer,
                  cvac_args=cvac_args)


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
