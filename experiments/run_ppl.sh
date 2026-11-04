#!/usr/bin/env bash
# Experiment 1 sweep. Each config is its own process; finished configs (json present) are skipped.
# Usage: bash experiments/run_ppl.sh [sinks|bf16|full|llama|v2|llama64|all]
set -u
cd "$(dirname "$0")"
PY=${PY:-../.venv/Scripts/python.exe}
run() {  # model config tokens [suffix]
  [ -f "../results/ppl/$1/$2${4:-}.json" ] && { echo "skip $1/$2"; return; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" ppl.py --model "$1" --config "$2" --tokens "$3" \
    2>&1 | grep -v -i -E "warn|Loading weights|Redirects" | tee -a "../results/ppl_log.txt"
}
what=${1:-all}
if [[ $what == sinks || $what == all ]]; then
  # main configs, 64K tokens: same total cache size (1,024)
  run qwen3-0.6b sink4_w1020 65536
  run qwen3-0.6b sink0_w1024 65536
  # ablations, 16K tokens
  for s in 0 1 2 8; do run qwen3-0.6b sink${s}_w1020 16384; done
  for w in 256 512 2048; do
    run qwen3-0.6b sink4_w$((w - 4)) 16384
    run qwen3-0.6b sink0_w$w 16384
  done
  run qwen3-0.6b sink4_w1020_norot 16384
fi
if [[ $what == bf16 || $what == all ]]; then
  # same sweep with keys stored in bf16: shows the accumulated re-rotation drift
  run qwen3-0.6b sink4_w1020_bf16k 65536
  run qwen3-0.6b sink0_w1024_bf16k 65536
  for s in 0 1 2 8; do run qwen3-0.6b sink${s}_w1020_bf16k 16384; done
  for w in 256 512 2048; do
    run qwen3-0.6b sink4_w$((w - 4))_bf16k 16384
    run qwen3-0.6b sink0_w${w}_bf16k 16384
  done
fi
if [[ $what == full || $what == all ]]; then
  run qwen3-0.6b full 65536         # until OOM
fi
if [[ $what == llama || $what == all ]]; then
  run llama3.2-1b sink4_w1020 16384
  run llama3.2-1b sink0_w1024 16384
  run llama3.2-1b full 16384
fi
if [[ $what == v2 || $what == all ]]; then
  # v2: sinks moved forward, window keys untouched, bf16 storage. Same configs as the fp32/bf16 v1
  # sweeps, so drift (cache size 256..2048), memory and speed compare directly.
  run qwen3-0.6b v2sink4_w1020 65536
  run qwen3-0.6b v2sink0_w1024 65536
  for w in 256 512 2048; do run qwen3-0.6b v2sink4_w$((w - 4)) 16384; done
  run llama3.2-1b v2sink4_w1020 16384
fi
if [[ $what == llama64 ]]; then
  # Llama-3.2-1B to 64K tokens (its trained range is 128K; 32 KiB/token, so the full cache fits)
  run llama3.2-1b full 65536 _64k
  run llama3.2-1b v2sink4_w1020 65536 _64k
  run llama3.2-1b v2sink0_w1024 65536 _64k
fi
