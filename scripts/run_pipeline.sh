#!/usr/bin/env bash
# Full benchmark pipeline on a GPU machine.
# Usage: bash scripts/run_pipeline.sh            (edit MODELS / READERS below first)
set -euo pipefail

OUT=${OUT:-runs/v0}
N_PER_CELL=${N_PER_CELL:-30}          # 15 cells x 30 = 450 items
SAMPLED=${SAMPLED:-10}                # samples for H_sampled (0 = first-token entropy only)

# label=model_id[|chat_template_kwargs_json]
MODELS=(
  "qwen2.5-7b=Qwen/Qwen2.5-7B-Instruct"
  "qwen2.5-1.5b=Qwen/Qwen2.5-1.5B-Instruct"
  "llama3.1-8b=meta-llama/Llama-3.1-8B-Instruct"
  "mistral-7b-v0.3=mistralai/Mistral-7B-Instruct-v0.3"
  "gemma2-9b=google/gemma-2-9b-it"
  'qwen3-8b=Qwen/Qwen3-8B|{"enable_thinking": false}'
)
# Main reader + a second one for rank-stability checks. Keep readers out of MODELS if you can.
READERS=(
  "reader-qwen2.5-14b=Qwen/Qwen2.5-14B-Instruct"
  "reader-llama3.1-8b=meta-llama/Llama-3.1-8B-Instruct"
)

mkdir -p "$OUT"
[ -f "$OUT/items.jsonl" ] || drybench items --out "$OUT/items.jsonl" --n-per-cell "$N_PER_CELL"

for entry in "${MODELS[@]}"; do
  label=${entry%%=*}; rest=${entry#*=}; model=${rest%%|*}
  ctk=""; [[ "$rest" == *"|"* ]] && ctk=${rest#*|}
  extra=(); [ -n "$ctk" ] && extra=(--chat-template-kwargs "$ctk")
  for variant in default concise; do
    drybench generate --items "$OUT/items.jsonl" --out "$OUT/responses.jsonl" \
      --backend hf --model "$model" --model-name "$label" --variant "$variant" \
      --dtype bf16 --batch-size 32 --max-new-tokens 768 "${extra[@]}"
  done
  drybench signals --items "$OUT/items.jsonl" --responses "$OUT/responses.jsonl" \
    --out "$OUT/signals.jsonl" --backend hf --model "$model" --model-name "$label" \
    --dtype bf16 --sampled "$SAMPLED" "${extra[@]}"
done

for entry in "${READERS[@]}"; do
  rlabel=${entry%%=*}; reader=${entry#*=}
  drybench score --items "$OUT/items.jsonl" --responses "$OUT/responses.jsonl" \
    --out "$OUT/scored.$rlabel.jsonl" --reader-backend hf --reader "$reader" \
    --reader-dtype bf16 --mode greedy --constraint per_probe --ablation-delta 0.05
  drybench analyze --items "$OUT/items.jsonl" --scored "$OUT/scored.$rlabel.jsonl" \
    --signals "$OUT/signals.jsonl" --out-dir "$OUT/report.$rlabel"
done
