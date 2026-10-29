#!/usr/bin/env bash
# Step 1 of direction 1: EoS identity vs. extra positions, and which attention paths matter.
# Usage:  bash run_step1.sh [GPU_ID]
# Needs:  the Breckner & Schuster repo cloned next to this folder (for the datasets):
#         git clone https://github.com/minaolensarah/thinking_eos-by-eos.git
set -euo pipefail
GPU=${1:-0}
MODEL=${MODEL:-GSAI-ML/LLaDA-1.5}
DATA=${DATA:-thinking_eos-by-eos/datasets}
OUT=${OUT:-results_step1}
PER_LEVEL=${PER_LEVEL:-40}
export CUDA_VISIBLE_DEVICES=$GPU

run() { python eos_ablation.py --model "$MODEL" --data_dir "$DATA" --out_dir "$OUT" --per_level "$PER_LEVEL" "$@"; }

# 0) Self-test: the attention patch must actually control attention. Stop if it fails.
run --task maths --selftest

for TASK in maths boxes sudoku; do
  # 1) Filler identity: committed EoS vs. never-committed [MASK], full attention.
  #    (n_tail=0 is the shared baseline; dot/random were already tested by Breckner & Schuster.)
  run --task $TASK --n_tail 0,8,32,64,128 --fillers eos,mask --attn full

  # 2) Attention-path ablations at a tail length where the effect is clear.
  run --task $TASK --n_tail 64 --fillers eos,mask \
      --attn noread,isolated,nowrite_A,nowrite_P,nowrite_all
done

# Optional replication of their other-filler appendix:
# for TASK in maths boxes sudoku; do run --task $TASK --n_tail 64 --fillers dot,random --attn full; done

python summarize.py --out_dir "$OUT"
