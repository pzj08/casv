# AORC Codex 修改提示词

下面内容可以直接粘贴给 Codex，用于在 WeSpeaker 官方 ResNet34 baseline 上实现 **AORC: Age-Ordered Residual Compensation for Cross-Age Speaker Verification**。

```text
你现在在一个 WeSpeaker 官方代码仓库中工作。我已经有官方 ResNet34 + AAM/ArcFace speaker verification baseline。请在保持 baseline 可复现、可关闭新模块的前提下，实现一个用于 Cross-Age Speaker Verification 的三模块方法：

方法暂名：AORC，Age-Ordered Residual Compensation for Cross-Age Speaker Verification。

核心目标：
在 WeSpeaker ResNet34 baseline 上，不更换 backbone，只增加训练阶段的年龄有序建模与测试阶段可用的年龄残差补偿，使最终 speaker embedding 对跨年龄变化更鲁棒。

请先阅读并定位当前仓库中的：
1. ResNet34 模型定义；
2. speaker embedding 输出位置；
3. AAM-Softmax / ArcFace / margin-based speaker loss 的实现；
4. dataset / dataloader 返回 utt、speaker label 的位置；
5. train loop 中 loss 计算和日志记录的位置；
6. embedding extraction / scoring 脚本中使用 embedding 的位置。

不要假设固定文件路径。请用 grep/search 先定位现有实现，再最小侵入式修改。

============================================================
一、总体结构
============================================================

现有 baseline：

输入语音特征 x_i，经 WeSpeaker ResNet34 得到 utterance-level 表征：

    h_i = F_theta(x_i)

baseline speaker embedding：

    e_i_raw = W_s h_i

baseline speaker loss：

    L_spk = L_AAM(z_i, s_i)

其中 s_i 是 speaker label。

请在此基础上增加三个主模块：

M1. OAM: Ordinal Age Manifold
    用年龄组标签学习有序年龄流形。

M2. ORC: Ordinal Residual Compensation
    根据预测的年龄分布估计 age residual，并从 raw speaker embedding 中补偿掉。

M3. CAA: Cross-Age Aggregation
    对同一说话人的跨年龄 positive pairs 做 age-gap-aware supervised contrastive learning。

最终 full model：

    h_i = F_theta(x_i)

    e_i_raw = W_s h_i

    z_i_age, q_i_age = OAM(h_i)

    r_i_age = ORC(q_i_age)

    z_i_spk = Normalize(e_i_raw - gamma * r_i_age)

    L = L_AAM(z_i_spk, s_i)
        + lambda_oam * L_OAM
        + lambda_caa * L_CAA
        + lambda_smooth * L_smooth

测试阶段：
不需要真实 age label。模型自己预测 q_i_age，然后进行 residual compensation，最终输出 z_i_spk 用于 cosine scoring / PLDA scoring。

当关闭新模块时：
必须完全保持原始 ResNet34 baseline 行为，不要求 age label，不改变 embedding extraction 和 scoring 结果。

============================================================
二、年龄标签与数据输入
============================================================

请为训练数据增加可选 age label 支持。

要求：
1. 支持配置项 age_label_file，格式可以是：
       utt_id age_group
   或：
       utt_id age_value

2. 如果文件给的是 age_group，则 age_group 必须是整数：
       g_i in {0, 1, ..., K-1}

3. 如果文件给的是真实或估计年龄 age_value，则根据配置 age_bins 转成 age_group。

例如：
    age_bins = [18, 26, 36, 46, 56, 66]

表示：
    group 0: age < 18
    group 1: 18 <= age < 26
    group 2: 26 <= age < 36
    group 3: 36 <= age < 46
    group 4: 46 <= age < 56
    group 5: 56 <= age < 66
    group 6: age >= 66

K = len(age_bins) + 1

也允许用户直接设置 num_age_groups=K 并提供已经离散好的 age_group。

4. dataloader 应返回：
       feats
       speaker_label
       age_group
   其中 age_group shape 为 [B]，dtype 为 torch.long。

5. 如果新模块启用但没有 age_label_file，应在训练开始时报错并说明缺少 age labels。

6. 如果新模块关闭，则不需要 age label，baseline 能照常训练。

7. 支持 ignore_index=-1。若某些 utt 没有年龄标签，则：
   - speaker loss 正常计算；
   - OAM / ORC / CAA 中涉及年龄监督的部分忽略这些样本；
   - ORC 在训练和测试时仍可使用模型预测年龄分布 q_i_age。

============================================================
三、模块 M1：OAM，Ordinal Age Manifold
============================================================

OAM 负责学习一个辅助 age subspace。

输入：
    h_i: [B, D_h]

输出：
    z_i_age: [B, D_age]
    q_i_age: [B, K]
    L_OAM

定义：

    z_i_age = Normalize(W_a h_i)

其中 W_a 是 age projection head，可以是 Linear 或 Linear-ReLU-Linear，维度由配置控制，默认 D_age = speaker embedding dim。

OAM 包含三个紧密相关的小目标，主消融中不要拆开，作为一个整体模块：

    L_OAM = L_ord + alpha_proxy * L_proxy + beta_dir * L_dir

------------------------------------------------------------
3.1 Ordinal age loss: L_ord
------------------------------------------------------------

不要使用普通 age classification 作为 full model 的年龄建模方式。请实现 cumulative ordinal regression / CORAL-style ordinal loss。

年龄组：

    g_i in {0, 1, ..., K-1}

定义 K-1 个有序二分类目标：

    r_{i,t} = 1[g_i > t],   t = 0, 1, ..., K-2

实现建议：
使用一个 ordinal head 输出一个 scalar score，再配合 K-1 个有序 thresholds。

    score_i = Linear(z_i_age)  -> shape [B, 1]

thresholds:
    raw_delta: learnable vector [K-1]
    delta = softplus(raw_delta) + eps
    thresholds = cumulative_sum(delta)

为了数值稳定，可以对 thresholds 做中心化：
    thresholds = thresholds - mean(thresholds)

rank logits:
    l_{i,t} = score_i - thresholds_t

rank probabilities:
    p_{i,t} = sigmoid(l_{i,t}) ≈ P(g_i > t)

ordinal BCE loss:

    L_ord = - mean_{i,t} [
        r_{i,t} log p_{i,t}
        + (1 - r_{i,t}) log(1 - p_{i,t})
    ]

请用 BCEWithLogitsLoss 实现，避免手动 log 导致数值不稳定。

从 cumulative probabilities 得到 age distribution q_i_age：

    q_{i,0} = 1 - p_{i,0}

    q_{i,k} = p_{i,k-1} - p_{i,k},     k = 1, ..., K-2

    q_{i,K-1} = p_{i,K-2}

由于 thresholds 有序，理论上 p_{i,t} 随 t 单调下降，因此 q_i_age 非负且 sum 接近 1。实现时仍需 clamp 到 eps 后重新 normalize，防止极端数值误差：

    q = clamp(q, min=eps)
    q = q / q.sum(dim=-1, keepdim=True)

逻辑检查：
- L_ord 让模型知道 age group 是有序标签，不是 nominal class。
- q_i_age 后续供 ORC 使用。
- 测试阶段不需要真实年龄，q_i_age 由模型预测。

------------------------------------------------------------
3.2 Ordinal prototype loss: L_proxy
------------------------------------------------------------

为每个年龄组设置一个可学习 prototype：

    P = {p_0, p_1, ..., p_{K-1}}

其中：
    p_k in R^{D_age}
    P shape: [K, D_age]

使用归一化 prototype：

    p_k = Normalize(p_k)

计算相似度：

    sim_{i,k} = cosine(z_i_age, p_k)

    logits_{i,k} = sim_{i,k} / tau_proxy

定义 age-distance negative weight：

    d_{i,k} = |g_i - k| / max(K - 1, 1)

    alpha_{i,k} = 1 + lambda_proto_dist * d_{i,k}

对正类 prototype，alpha 可以保留为 1，因为 d=0 时 alpha=1。

weighted prototype CE：

    L_proxy_i =
        - logits_{i,g_i}
        + logsumexp_k( logits_{i,k} + log(alpha_{i,k}) )

    L_proxy = mean_i L_proxy_i

只对 age_group 有效的样本计算。

逻辑检查：
- 离真实年龄组越远的 negative prototype 权重越大。
- 这会鼓励 age subspace 中不同年龄组按年龄距离拉开。
- 它不是普通 proxy loss，而是 ordinal-distance-aware proxy loss。

------------------------------------------------------------
3.3 Speaker-conditioned direction loss: L_dir
------------------------------------------------------------

只对同一 mini-batch 中同一说话人、不同年龄组的样本对计算方向一致性。

有效 pair 条件：

    s_i == s_j
    g_i != g_j
    age_group_i 和 age_group_j 都有效

若 g_i < g_j，则：

    v_age_ij =
        Normalize(z_j_age - z_i_age)

    v_proto_ij =
        Normalize(p_{g_j} - p_{g_i})

若 g_i > g_j，则交换 i,j，保证方向从年轻到年长。

direction consistency loss：

    L_dir =
        mean_{(i,j)} omega_ij * [1 - cosine(v_age_ij, v_proto_ij)]

其中 age-gap weight：

    omega_ij = 1 + beta_gap * |g_i - g_j| / max(K - 1, 1)

如果 batch 内没有有效 pair，则 L_dir = 0，并且不要报错。

实现注意：
- B 通常不大，O(B^2) pair mask 可以接受。
- 如果 batch 很大，可支持 max_dir_pairs 随机采样，默认 None。
- Normalize 前加 eps，避免除零。
- 该 loss 只作用于 age subspace 和 prototypes，不应直接拉动 speaker space 按年龄排列。

逻辑检查：
- 语音中不能用任意不同说话人的样本建模年龄方向，因为 timbre、性别、口音、信道会污染 age direction。
- 同一说话人的跨年龄 pair 更接近真实 age drift。
- 这是 speaker-conditioned age trajectory，而不是全局 age direction。

============================================================
四、模块 M2：ORC，Ordinal Residual Compensation
============================================================

ORC 是核心创新模块。

目标：
根据 OAM 预测出的年龄分布 q_i_age，估计一个 age residual，然后从 raw speaker embedding 中扣除。

定义 K 个可学习 age residual basis：

    B = {b_0, b_1, ..., b_{K-1}}

其中：
    b_k in R^{D_spk}
    B shape: [K, D_spk]

年龄残差：

    r_i_age = sum_{k=0}^{K-1} q_{i,k} * b_k

补偿后的 speaker embedding：

    z_i_spk = Normalize(e_i_raw - gamma * r_i_age)

其中：
- gamma 可以是配置中的常数 residual_scale；
- 也可以实现为可学习标量 learnable_residual_scale；
- 默认建议 residual_scale = 1.0；
- 若使用 learnable gamma，建议初始化为 0.1 或 1.0，并允许配置。

平滑约束：

    L_smooth =
        sum_{k=0}^{K-2} || b_{k+1} - b_k ||_2^2

建议使用 mean 而不是 sum，避免 K 改变时 loss scale 变化太大：

    L_smooth = mean_k || b_{k+1} - b_k ||_2^2

speaker loss 应作用在补偿后的 embedding 上：

    L_spk = L_AAM(z_i_spk, s_i)

而不是 raw embedding。

ORC 的梯度设计：
请添加配置项：

    detach_age_prob_for_residual: true/false

默认建议 true：

    q_for_residual = q_i_age.detach()

这样 speaker loss 不会直接把 age head 拉偏，age head 主要由 OAM 学习。  
如果设为 false，则 ORC 完全端到端，speaker loss 也能影响 age distribution。

逻辑检查：
- ORC 不是简单强行删除年龄信息，而是显式估计 age-ordered residual。
- age residual basis 按年龄组排列，并通过 L_smooth 保持有序平滑。
- 测试阶段可以直接使用预测的 q_i_age 做补偿，不需要真实年龄标签。
- ORC 依赖 OAM 的 q_i_age；如果做 “w/o OAM” 消融，可用普通 age CE 的 softmax q_i_age 替代。

============================================================
五、模块 M3：CAA，Cross-Age Aggregation
============================================================

CAA 作用在最终 speaker embedding z_i_spk 上，用于直接拉近同一说话人的跨年龄正样本。

使用 supervised contrastive loss 的 age-gap-aware 版本。

输入：
    z_spk: [B, D_spk], normalized
    speaker labels s: [B]
    age groups g: [B]

相似度：

    sim_{i,a} = cosine(z_i_spk, z_a_spk) / tau_caa

positive set:

    P(i) = { p | p != i and s_p == s_i }

age-gap positive weight：

    eta_{i,p} = 1 + gamma_caa * |g_i - g_p| / max(K - 1, 1)

如果 g_i 或 g_p 无效，则 eta_{i,p} = 1，但仍可作为同 speaker positive。

loss：

    L_CAA =
        - mean_i mean_{p in P(i)}
            eta_{i,p} *
            log [
                exp(sim_{i,p})
                /
                sum_{a != i} exp(sim_{i,a})
            ]

实现要点：
- 使用 logsumexp 实现 denominator，mask 掉 self。
- 如果一个 anchor 没有 positive，则跳过该 anchor。
- 如果整个 batch 没有 positive pair，则 L_CAA = 0，不报错。
- 建议对 eta 做 batch 内归一化或保持均值接近 1，防止 loss scale 随 age gap 变化过大。可选：
      eta = eta / eta.mean().detach()
- CAA 不替代 AAM loss，只是辅助提升跨年龄 positive 聚合。

逻辑检查：
- AAM 保证 speaker discrimination。
- CAA 显式关注同一说话人大年龄差 pair。
- 这与 ORC 互补：ORC 做 residual compensation，CAA 直接优化 speaker space 中跨年龄正样本的紧致性。

============================================================
六、总损失与训练逻辑
============================================================

full model 总损失：

    L_total =
        L_AAM(z_spk, s)
        + lambda_oam * L_OAM
        + lambda_caa * L_CAA
        + lambda_smooth * L_smooth

其中：

    L_OAM = L_ord + alpha_proxy * L_proxy + beta_dir * L_dir

展开后：

    L_total =
        L_AAM(z_spk, s)
        + lambda_oam * (
              L_ord
              + alpha_proxy * L_proxy
              + beta_dir * L_dir
          )
        + lambda_caa * L_CAA
        + lambda_smooth * L_smooth

请确保所有 loss 都是标量 tensor，且 device 正确。

默认超参数建议全部写入配置文件，不要硬编码：

    num_age_groups: 7
    age_bins: null
    age_label_file: null
    age_label_type: group  # group or value

    enable_oam: true
    enable_orc: true
    enable_caa: true

    age_mode: ordinal  # ordinal or ce
    age_emb_dim: same_as_spk_emb_dim

    lambda_oam: 0.1
    alpha_proxy: 0.1
    beta_dir: 0.05
    lambda_caa: 0.05
    lambda_smooth: 1.0e-3

    tau_proxy: 0.1
    tau_caa: 0.07

    lambda_proto_dist: 1.0
    beta_gap: 1.0
    gamma_caa: 1.0

    residual_scale: 1.0
    learnable_residual_scale: false
    detach_age_prob_for_residual: true

    ignore_age_index: -1
    max_dir_pairs: null

训练逻辑：

1. 如果 enable_oam=false, enable_orc=false, enable_caa=false：
       完全走 baseline。

2. 如果 enable_oam=true：
       需要 age labels。
       计算 z_age, q_age, L_OAM。

3. 如果 enable_orc=true：
       需要 q_age。
       使用 q_age 生成 r_age。
       z_spk = Normalize(e_raw - residual_scale * r_age)。
       L_AAM 使用 z_spk。
       计算 L_smooth。

4. 如果 enable_orc=false：
       z_spk = Normalize(e_raw) 或使用 baseline 原始 embedding 逻辑。
       必须尽量保持 baseline 行为。

5. 如果 enable_caa=true：
       用 z_spk 计算 L_CAA。

6. 日志中至少记录：
       loss_total
       loss_spk
       loss_oam
       loss_ord
       loss_proxy
       loss_dir
       loss_caa
       loss_smooth
       residual_scale/gamma
       age_acc 或 age_mae/ordinal_acc，可选

============================================================
七、ordinary age CE 对照模式
============================================================

为了做 “w/o OAM” 或 “Age CE baseline”，请支持 age_mode=ce。

当 age_mode=ce 时：

    age_logits = AgeClassifier(z_age) -> [B, K]
    L_age_ce = CrossEntropy(age_logits, g)
    q_age = softmax(age_logits)

此时：
- 不使用 L_ord；
- 不使用 ordinal thresholds；
- 可选择不使用 L_proxy 和 L_dir；
- ORC 仍可以用 q_age 做 residual compensation。

这样可以实现以下重要消融：

1. Baseline:
       enable_oam=false, enable_orc=false, enable_caa=false

2. Age CE:
       age_mode=ce, enable_oam=true, enable_orc=false, enable_caa=false

3. OAM:
       age_mode=ordinal, enable_oam=true, enable_orc=false, enable_caa=false

4. OAM + ORC:
       age_mode=ordinal, enable_oam=true, enable_orc=true, enable_caa=false

5. Full:
       age_mode=ordinal, enable_oam=true, enable_orc=true, enable_caa=true

6. Full w/o OAM:
       age_mode=ce, enable_oam=true, enable_orc=true, enable_caa=true
   解释：把 ordinal age manifold 替换成普通 age CE，验证 OAM 的必要性。

7. Full w/o ORC:
       age_mode=ordinal, enable_oam=true, enable_orc=false, enable_caa=true

8. Full w/o CAA:
       age_mode=ordinal, enable_oam=true, enable_orc=true, enable_caa=false

============================================================
八、消融实验配置文件
============================================================

请新增或复制现有 ResNet34 训练 yaml，创建以下配置模板：

1. resnet34_aam_baseline.yaml
   - 完全等价官方 baseline。

2. resnet34_aam_age_ce.yaml
   - baseline + ordinary age classification。
   - enable_oam=true
   - age_mode=ce
   - enable_orc=false
   - enable_caa=false

3. resnet34_aam_oam.yaml
   - baseline + OAM。
   - enable_oam=true
   - age_mode=ordinal
   - enable_orc=false
   - enable_caa=false

4. resnet34_aam_oam_orc.yaml
   - baseline + OAM + ORC。
   - enable_oam=true
   - age_mode=ordinal
   - enable_orc=true
   - enable_caa=false

5. resnet34_aam_aorc_full.yaml
   - full model。
   - enable_oam=true
   - age_mode=ordinal
   - enable_orc=true
   - enable_caa=true

6. resnet34_aam_full_wo_oam.yaml
   - full w/o OAM。
   - age_mode=ce
   - enable_oam=true
   - enable_orc=true
   - enable_caa=true

7. resnet34_aam_full_wo_orc.yaml
   - full w/o ORC。
   - age_mode=ordinal
   - enable_oam=true
   - enable_orc=false
   - enable_caa=true

8. resnet34_aam_full_wo_caa.yaml
   - full w/o CAA。
   - age_mode=ordinal
   - enable_oam=true
   - enable_orc=true
   - enable_caa=false

============================================================
九、代码组织建议
============================================================

请尽量新增文件，而不是大幅改动原始代码。

建议新增：

1. losses / aorc_losses.py
   包含：
       OrdinalAgeLoss
       OrdinalPrototypeLoss
       SpeakerConditionedDirectionLoss
       CrossAgeAggregationLoss

2. models / aorc_modules.py
   包含：
       OrdinalAgeHead
       AgeResidualCompensation
       AORCWrapper 或者 AORCMixin

3. dataset 相关文件中增加 age_label_file 读取逻辑。

4. train loop 中增加：
       - 从 batch 读取 age_group；
       - 调用模型 forward 返回 dict；
       - 计算额外 losses；
       - 汇总 total loss；
       - logging。

模型 forward 建议返回 dict：

    outputs = {
        "embedding": z_spk,          # 用于 speaker loss 和 extraction
        "raw_embedding": e_raw,
        "age_embedding": z_age,
        "age_distribution": q_age,
        "age_logits": age_logits or rank_logits,
        "residual": r_age,
        "extra_losses": {
            "loss_ord": ...,
            "loss_proxy": ...,
            "loss_dir": ...,
            "loss_smooth": ...
        }
    }

如果当前 WeSpeaker 的训练代码只接受 tensor embedding，则请保持兼容：
- baseline forward 仍可返回 tensor；
- 启用 AORC 时返回 dict；
- train loop 判断输出类型。

或者新增参数：
    return_dict=True/False

embedding extraction 阶段：
- 默认使用 outputs["embedding"]。
- 如果模型返回 tensor，则沿用原逻辑。
- 不要求 age_group 输入。
- 确保模型 eval() 时也会预测 q_age 并补偿。

============================================================
十、数值稳定性与合理性检查
============================================================

请实现以下保护：

1. 所有 Normalize 使用 eps，例如 F.normalize(x, p=2, dim=-1, eps=1e-12)。

2. q_age 由 ordinal probabilities 转换后：
       clamp(min=eps)
       normalize sum to 1

3. prototype 和 residual basis 初始化：
       prototypes: small normal, then normalized in forward
       residual basis: zeros or small normal
   建议 residual basis 初始化为 zeros 或 std=0.01，避免训练初期破坏 baseline embedding。

4. 如果 no valid age labels：
       L_ord, L_proxy, L_dir = 0
       但 speaker loss 正常。

5. 如果 no same-speaker cross-age pairs：
       L_dir = 0
       L_CAA 对无 positive anchor 自动跳过。

6. L_CAA 的 denominator 必须 mask self。

7. age-gap weights 建议归一化或限制 scale：
       eta = eta / eta.mean().detach()
   防止大 age gap 造成 loss 爆炸。

8. 保证 enable_new_modules=false 时：
       不读取 age_label_file；
       不创建额外 loss；
       训练速度和 baseline 基本一致；
       可加载旧 checkpoint。

9. 如果加载 baseline checkpoint 到 AORC 模型：
       允许 strict=False；
       新增模块随机初始化；
       已有 ResNet34 和 speaker classifier 权重正常加载。

============================================================
十一、单元测试 / smoke test
============================================================

请增加或至少本地运行以下最小测试：

1. OrdinalAgeHead test:
   输入 random h: [8, D]
   输出：
       z_age shape [8, D_age]
       q_age shape [8, K]
       q_age 每行 sum 接近 1
       q_age 没有 NaN

2. Ordinal loss test:
   g in [0, K-1]
   L_ord finite。

3. Prototype loss test:
   L_proxy finite。

4. Direction loss test:
   - batch 中有同 speaker 不同 age pair 时 finite；
   - 没有 pair 时返回 0 tensor。

5. ORC test:
   e_raw shape [8, D]
   q_age shape [8, K]
   z_spk shape [8, D]
   L_smooth finite。

6. CAA test:
   - 有 positive pair 时 finite；
   - 没有 positive pair 时返回 0 tensor。

7. Baseline compatibility test:
   enable_oam=false, enable_orc=false, enable_caa=false 时，forward 输出和原 baseline 兼容。

8. Inference test:
   eval mode 下不传 age_group，模型仍能输出 embedding。

============================================================
十二、实验表目标
============================================================

最终代码应支持下面的主消融表：

Table A: main ablation

    Method                 OAM   ORC   CAA
    ResNet34 + AAM          -     -     -
    + Age CE                -     -     -
    + OAM                   ✓     -     -
    + OAM + ORC             ✓     ✓     -
    + OAM + ORC + CAA       ✓     ✓     ✓

Table B: removal ablation

    Method                 OAM   ORC   CAA
    Full w/o OAM            CE    ✓     ✓
    Full w/o ORC            ✓     -     ✓
    Full w/o CAA            ✓     ✓     -
    Full                    ✓     ✓     ✓

请确保配置文件可以直接跑出这些实验。

============================================================
十三、请完成的具体开发任务
============================================================

请按以下顺序执行：

1. Inspect repository:
   - 找到 ResNet34 模型、speaker embedding、AAM loss、dataset、train loop、embedding extraction。
   - 简要总结需要改哪些文件。

2. Implement age label loading:
   - 支持 utt_id 到 age_group / age_value。
   - dataloader batch 返回 age_group。
   - baseline 关闭新模块时不受影响。

3. Implement OAM:
   - OrdinalAgeHead。
   - CORAL-style ordinal loss。
   - q_age distribution conversion。
   - ordinal prototype loss。
   - speaker-conditioned direction loss。

4. Implement ORC:
   - residual basis。
   - residual compensation。
   - smoothness loss。
   - test-time compensation without ground-truth age。

5. Implement CAA:
   - age-gap-aware supervised contrastive loss。
   - robust handling of no positive pairs。

6. Integrate into training:
   - total loss aggregation。
   - logging。
   - config-driven enable flags。
   - keep baseline unchanged when disabled。

7. Integrate into embedding extraction:
   - use compensated embedding outputs["embedding"] when AORC enabled。
   - no age label needed during extraction。

8. Add config templates:
   - baseline
   - age CE
   - OAM
   - OAM+ORC
   - full
   - full w/o OAM
   - full w/o ORC
   - full w/o CAA

9. Add minimal tests or smoke test script:
   - verify all losses finite。
   - verify forward and inference work。
   - verify baseline mode still works。

10. After implementation:
   - Show a concise diff summary。
   - Show example training commands。
   - Show example extraction/scoring commands。
   - Show all new config keys and defaults。
```
