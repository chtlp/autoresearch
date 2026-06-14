# 深度优化分析报告：验证比特率 (Val BPB) 与 MFU 的双重突破

本报告详细梳理并分析了从初始 Baseline（`1.144` BPB, ~12.4% MFU）到当前最优配置（`1.012` BPB, ~24.1% MFU）的核心改进路线与底层硬件/算法层面的技术机理。

---

## 核心改进路线与性能概览（完整历程）

| 阶段 | 核心改动 | Val BPB | MFU | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **原始 Baseline** | 默认参数（`2**19` batch, `"SSSL"` 窗口）| `1.144210` | `~12.4%` | 起点 |
| **第一步** | 减半 Batch Size to `2**18` | `1.074104` | `~12.4%` | 统计收敛优化 |
| **第二步** | 注意力改为 `"LLLL"` 全局模式 | `1.022657` | `~24.8%` | **硬件加速 + 感受野** |
| **第三步** | LR 退火调整（WARMDOWN=0.6, FINAL=0.02）| `1.020362` | `~24.8%` | 学习率调度 |
| **dev 分支最优** | EMBEDDING_LR=0.5 | `1.020096` | `~24.8%` | 昨日最优 |
| **autoresearch 基线** | 同上配置作为今日起点 | `1.020871` | `~24%` | 今日起点 |
| **今日最优** | eps=1e-8, MLP 5x, MATRIX_LR=0.03 | **`1.012253`** | `~24.1%` | **当前最优** |

---

## 第一部分：早期优化（dev 分支）

### 📊 阶段对比表

| 阶段 | 核心改动 | 验证比特率 (Val BPB) | 计算吞吐率 (MFU) | 优化类型 |
| :--- | :--- | :--- | :--- | :--- |
| **初始 Baseline** | 默认参数 (`2**19` batch size, `"SSSL"` 窗口) | `1.144210` | `~12.4%` | 基准线 |
| **第一步** | 减半 Batch Size to `2**18` | `1.074104` | `~12.4%` | 统计收敛优化 |
| **第二步** | 注意力机制变更为全全局模式 `"LLLL"` | `1.022657` | `~24.8%` | **硬件加速 + 感受野优化** |
| **第三步** | 调整 LR 退火退火区间与精细度 (0.6 / 0.02) | `1.020362` | `~24.8%` | 学习率调度优化 |
| **第四步 (dev 最优)** | 锁定 Embedding 层 LR 至 `0.5` | **`1.020096`** | **`24.82%`** | 稀疏表征优化 |

### 🚀 1. 突破硬件限制，打通原生加速通道 (MFU 提升核心)

* **具体改动**：将注意力机制的滑动窗口模式 `WINDOW_PATTERN` 从 `"SSSL"` 变更为 `"LLLL"` (全全局注意力)。
* **BPB 贡献**：`1.074` ➡️ `1.022` (降低约 `0.052`)
* **MFU 贡献**：`~12.4%` ➡️ `~24.8%` (**相对提升 100%**)

> [!IMPORTANT]
> **技术原理**：在当前 Blackwell (SM120) 硬件平台上，由于缺乏预编译的 FA3 算子，代码退回到 PyTorch SDPA。
> * **"SSSL"**：滑动窗口需要自定义 Causal Mask，阻断编译器融合优化，MFU 极低 (~12.4%)。
> * **"LLLL"**：全局注意力通过 `is_causal=True` 匹配原生 FlashAttention 路径，吞吐量直接翻倍，5分钟内步数从 500 增至 1025。

### 📈 2. 减小批次大小，提升优化步频

* **BPB 贡献**：`1.144` ➡️ `1.074` (降低约 `0.070`)

> [!TIP]
> 在 5 分钟训练预算下，更多步数优于更大批次。Batch 减半 → 步数翻倍 → 更充分的参数探索。

### 📉 3. 优化学习率衰减曲线

* WARMDOWN_RATIO: `0.5` → `0.6`；FINAL_LR_FRAC: `0.0` → `0.02`
* **BPB 贡献**：`1.022` ➡️ `1.0203`

### 🧬 4. 解耦并精调 Embedding 层学习率

* EMBEDDING_LR: `0.6` → `0.5`
* **BPB 贡献**：`1.020362` ➡️ `1.020096`

> [!NOTE]
> Embedding 更新极稀疏（每步只有出现的 Token 有梯度），需要独立于 Muon 矩阵参数的学习率调控。

---

## 第二部分：今日自动调优（autoresearch/jun14b 分支）

今日起点：`1.020871`（配置同 dev 分支最优）  
今日终点：`1.012253`，累计提升 **0.0086 BPB**

### 今日关键里程碑

| 改动 | val_bpb | 提升 |
| :--- | :--- | :--- |
| 起点基线 | 1.020871 | — |
| 减半 batch（BATCH=64, TOTAL=2^17）| 1.019782 | -0.001 |
| MLP 4x → 5x（甜点）| 1.015941 | -0.004 |
| WEIGHT_DECAY 0.2 → 0.12 | 1.014654 | -0.001 |
| MATRIX_LR 0.026 → 0.029 | 1.014170 | -0.0005 |
| **Adam eps 1e-10 → 1e-8** | **1.012544** | **-0.0016** |
| MATRIX_LR 0.029 → 0.03（+eps=1e-8）| **1.012253** | -0.0003 |

### 完整实验记录

#### 阶段 A：初始超参探索

| commit | val_bpb | status | 描述 |
| :--- | :--- | :--- | :--- |
| `40e9348` | 1.020871 | keep | 今日基线 |
| `5a76ac2` | 1.020823 | keep | FINAL_LR_FRAC 0.02→0.01 |
| `c1a9500` | 1.021484 | discard | WARMDOWN_RATIO 0.6→0.5 |
| `88923ab` | 1.021417 | discard | WEIGHT_DECAY 0.2→0.1 |
| `b288d4c` | 1.023760 | discard | ADAM_BETAS (0.8→0.9) |

#### 阶段 B：Batch Size 实验

| commit | val_bpb | status | 描述 |
| :--- | :--- | :--- | :--- |
| `ed547f6` | 1.019782 | keep | DEVICE_BATCH=64, TOTAL=2^17（2倍步数）|
| `b3185b0` | 1.032758 | discard | DEVICE_BATCH=32（太小）|
| `55eb45a` | 1.028096 | discard | DEPTH=10 + BATCH=64 |
| `386d770` | 1.040523 | discard | DEPTH=10（66GB，收敛慢）|

#### 阶段 C：MATRIX_LR / EMBEDDING_LR 调优

| commit | val_bpb | status | 描述 |
| :--- | :--- | :--- | :--- |
| `c007d8d` | 1.018386 | keep | MATRIX_LR 0.04→0.03 |
| `53c6fa7` | 1.018289 | keep | MATRIX_LR 0.03→0.025 |
| `7b5f817` | 1.017874 | keep | MATRIX_LR 0.025→0.02 |
| `4991686` | 1.019549 | discard | MATRIX_LR 0.02→0.015（过低）|
| `60184f3` | 1.018154 | discard | EMBEDDING_LR 0.5→0.4 |
| `12e03a4` | 1.017158 | keep | EMBEDDING_LR 0.5→0.6 |
| `d45250e` | 1.017760 | discard | EMBEDDING_LR 0.6→0.7 |

#### 阶段 D：广泛架构与调度探索（均 discard）

| commit | val_bpb | 描述 |
| :--- | :--- | :--- |
| `00fe112` | 1.060443 | WINDOW LLLL→SSSL（**大幅变差**）|
| `3cbbeea` | 1.059768 | DEPTH 8→6（容量不足）|
| `386d770` | 1.040523 | DEPTH 8→10（收敛慢）|
| `57c7fb9` | 1.023329 | SwiGLU MLP（ReLU² 更好）|
| `bb9f3de` | 1.023332 | GQA n_kv_head=2 |
| `a86053d` | 1.022778 | softcap 15→30 |
| `0f7c3f0` | 1.024470 | HEAD_DIM 128→64 |
| `730be81` | 1.025508 | ASPECT_RATIO 64→80 |
| `e84dffd` | 1.020147 | cosine warmdown（线性更好）|
| `fe36dc0` | 1.020526 | WARMUP_RATIO 0.0→0.05 |
| `1bd20d5` | 1.019766 | softcap 15→10 |
| `b1fca87` | 1.017579 | FINAL_LR_FRAC 0.01→0.005 |
| `1ecb76a` | 1.017473 | WARMDOWN_RATIO 0.6→0.7 |
| `6176df7` | 1.018382 | WEIGHT_DECAY 0.2→0.3 |
| `d7fd199` | 1.018464 | ADAM_BETAS (0.8,0.99) |
| `7f6818f` | 1.017810 | UNEMBEDDING_LR 0.008 |
| `e8c833c` | 1.017686 | Muon beta2 0.95→0.99 |
| `37fd4bf` | 1.017956 | Muon momentum warmup 300→600 |
| `227fd5d` | 1.018198 | SCALAR_LR 0.25 |
| `24a8489` | 1.019397 | SCALAR_LR 1.0 |

#### 阶段 E：MLP 扩展比例探索（重大突破）

| commit | val_bpb | status | 描述 |
| :--- | :--- | :--- | :--- |
| `c2a48af` | 1.020439 | discard | MLP 4x→8x（太慢）|
| `8662daf` | 1.016843 | keep | MLP 4x→6x |
| `ced0745` | **1.015941** | keep | MLP 6x→**5x**（甜点，24.2GB）|
| `8d5257f` | 1.016044 | discard | EMBEDDING_LR 0.7 with MLP 5x |

#### 阶段 F：MLP 5x 基础上精调

| commit | val_bpb | status | 描述 |
| :--- | :--- | :--- | :--- |
| `cda95dd` | 1.015409 | keep | MATRIX_LR 0.024→0.026 |
| `cbf3271` | 1.014946 | keep | WEIGHT_DECAY 0.2→0.15 |
| `080c325` | 1.014654 | keep | WEIGHT_DECAY 0.15→**0.12** |
| `3b6de27` | 1.014734 | discard | WEIGHT_DECAY 0.12→0.11 |
| `1f62673` | 1.014634 | keep | EMBEDDING_LR 0.6→**0.55** |
| `54d8dfa` | 1.014170 | keep | MATRIX_LR 0.027→**0.029** |
| `07188fd` | 1.014491 | discard | MATRIX_LR 0.029→0.030 |
| `6b9c5a6` | 1.026876 | discard | **禁用 value embeddings（至关重要！）** |
| `5e8bc9e` | 1.016619 | discard | DEPTH 8→7 |
| `0e1814f` | 1.014247 | discard | gradient clipping max_norm=1.0 |
| `99cee3a` | 1.019532 | discard | x0_lambdas init 0.1→0.0 |
| `ed9755d` | 1.012531 | discard | x0_lambdas init 0.1→0.2 |
| `d7b081f` | 1.016098 | discard | VE 所有层（交替更好）|

#### 阶段 G：Adam eps（最大单次突破）

| commit | val_bpb | status | 描述 |
| :--- | :--- | :--- | :--- |
| `94261e8` | **1.012544** | keep | Adam eps 1e-10→**1e-8**（+0.0016！）|
| `603055d` | 1.014528 | discard | eps 1e-8→1e-7（过大）|
| `a0be6a1` | **1.012253** | keep | MATRIX_LR 0.029→**0.03**（eps=1e-8）← **全局最优** |
| `0ee94b1` | 1.013498 | discard | MATRIX_LR 0.032 |
| `d8aa672` | 1.012688 | discard | EMBEDDING_LR 0.6 |
| `5f022ef` | 1.013350 | discard | WEIGHT_DECAY 0.10 |
| `ef10765` | 1.012979 | discard | MLP 6x（5x 仍更好）|
| `6aad41c` | 1.016360 | discard | UNEMBEDDING_LR 0.002（大幅变差）|
| `bcaae7a` | 1.014315 | discard | ADAM_BETAS (0.85,0.95) |

---

## 当前最优配置

```
DEVICE_BATCH_SIZE  = 64
TOTAL_BATCH_SIZE   = 2^17
MLP_EXPANSION      = 5x
MATRIX_LR          = 0.03
EMBEDDING_LR       = 0.55
WEIGHT_DECAY       = 0.12
ADAM_EPS           = 1e-8
FINAL_LR_FRAC      = 0.01
WARMDOWN_RATIO     = 0.6
WINDOW_PATTERN     = LLLL
```

val_bpb = **1.012253** | memory = 24.2 GB | MFU ≈ 24.1%  
（相比原始基线 1.144210，累计提升 **0.1319 BPB**）

---

## 关键规律总结

1. **WINDOW_PATTERN = LLLL 不可更改**：改回 SSSL 损失暴增 +0.04，同时 MFU 腰斩。
2. **Value Embeddings 不可禁用**：禁用后损失增加 +0.01，是架构核心组件。
3. **MLP 5x 是最优扩展比**：4x 容量不足，6x 收益有限，8x 太慢。
4. **Adam eps = 1e-8**：今日最大单步突破（+0.0016），eps 过小（1e-10）造成数值不稳定。
5. **Batch = 64（TOTAL=2^17）**：在5分钟预算下，更多步数优于更大批次。
6. **DEPTH = 8 是最优深度**：6、7层容量不足，10层收敛太慢。
