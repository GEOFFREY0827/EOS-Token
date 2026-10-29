# EoS ablation — 方向 1 第一步

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GEOFFREY0827/EOS-Token/blob/main/EOS_ablation_colab.ipynb)

在 LLaDA 上回答两个问题，全部只需推理，不需要训练。

1. **EoS 带来的提升，来自 EoS 这个 token 本身，还是来自"多出来的位置"？**
   对比尾部是已提交的 EoS（`eos`，复现 Breckner & Schuster），还是一直保持 `[MASK]`、永不提交（`mask`，新条件）。
   点号、空格、随机 token 等其他填充物，原论文附录已经测过，这里只作为可选复现。
2. **提升是通过哪条注意力通路实现的？** 记区域 P = prompt，A = 答案槽位，T = 尾部。

| `--attn` | 屏蔽 | 回答的问题 |
|---|---|---|
| `full` | 无 | 对照组 |
| `noread` | T ↛ P | 尾部读不到题目时，还有没有提升？没有 → 确实在做计算 |
| `isolated` | T ↛ P、T ↛ A | 尾部只能看到自己，与输入无关。仍有提升 → 是 **sink / 偏置** 效应，不是计算 |
| `nowrite_A` | A ↛ T | 切断"尾部 → 答案"的直接通路 |
| `nowrite_P` | P ↛ T | 切断"尾部 → prompt → 答案"的间接通路 |
| `nowrite_all` | A ↛ T、P ↛ T | 理论上必须等于 `n_tail=0`，作为 sanity check |

## 一个必须知道的实现细节

LLaDA 官方建模代码在调用 SDPA 时写死了 `attn_mask=None`，所以传给 `model(...)` 的 `attention_mask` / `attention_bias` **会被忽略**（原论文代码传的 padding mask 实际上也没生效）。本脚本改为 hook 每个 block 的 `_scaled_dot_product_attention`，注入自己的布尔 mask，同时会屏蔽 left padding。

**跑任何结果之前先跑 `--selftest`。** 它检查两件事：`nowrite_all` 下答案槽位的 logits 应当和 `n_tail=0` 基本一致；`full` 下应当明显不同。显示 FAIL 就说明 hook 没生效，所有 attention 消融的结果都不可信。如果 bf16 噪声导致误判，可以加 `--dtype fp32` 再跑一次 selftest。

## 运行

**Colab（A100）**：打开 `EOS_ablation_colab.ipynb`，按顺序运行即可，结果保存到 Google Drive。

**自己的服务器**：

```bash
git clone https://github.com/minaolensarah/thinking_eos-by-eos.git   # 只用它的数据集
# 把本文件夹的三个文件放在同一目录
bash run_step1.sh 0          # 参数是 GPU 编号
```

或者单独跑某一组条件：

```bash
python eos_ablation.py --task maths --selftest
python eos_ablation.py --task maths --n_tail 0,8,32,64 --fillers eos,mask --attn full
python eos_ablation.py --task boxes --n_tail 64 --fillers eos --attn isolated,noread
python summarize.py
```

- 默认模型 `GSAI-ML/LLaDA-1.5`（原论文用的就是它），可用 `MODEL=GSAI-ML/LLaDA-8B-Instruct` 替换
- 默认每个难度取 40 题：maths 200 题、boxes 520 题、sudoku 480 题。`PER_LEVEL=0` 表示全量
- 答案槽位数 k 沿用原论文：maths 12、boxes 22、sudoku 19，去噪步数 = k，单 block，温度 0
- 结果写到 `results_step1/<task>/*.jsonl`（逐题），汇总在 `results_step1/summary.jsonl`；已跑完的条件会自动跳过，可断点续跑
- `summarize.py` 输出两张表（填充物 × 尾部长度、注意力消融），并为每个任务画一张图

单张 A100 上整套 `run_step1.sh` 大约需要两三个小时（粗略估计，没有实测）。显存不够就调小 `--batch_size`。

## 结果怎么读

- `mask ≈ eos`：收益来自"位置"，EoS 本身不重要，可以直接推进专用 slot 设计（方向 1 第二步）
- `mask << eos`：已提交的 EoS 嵌入起了关键作用，转向方向 3，分析它在机制上的作用
- `isolated ≈ full`：尾部只充当偏置或 sink，"隐式推理"这个解读需要打折扣。这会是一个重要的反例
- `noread << full` 且 `isolated << full`：尾部确实在读题并参与计算
- `nowrite_A` 和 `nowrite_P` 对比：看结果是直接写回答案，还是绕道 prompt 位置写回

## 已知差异

- 这里会屏蔽 padding，原论文的代码不会（见上文），所以 baseline 数值可能和论文略有出入。所有条件都在同一套代码下比较，因此条件之间的对比是公平的
- 本环境无法安装 torch，torch 相关部分只做了静态检查。数据加载、prompt 构造和打分逻辑已用论文数据集测试通过。请以服务器上的 `--selftest` 结果为准
