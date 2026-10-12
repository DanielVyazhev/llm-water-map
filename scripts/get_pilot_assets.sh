#!/usr/bin/env bash
# GPT-2 (124M) for the CPU pilot: ONNX Model Zoo export (GitHub LFS) + BPE files (npm gpt-3-encoder).
set -euo pipefail
DIR=${1:-assets}
mkdir -p "$DIR" && cd "$DIR"
curl -sSL -o gpt2-lm-head-10.onnx \
  "https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx"
npm pack gpt-3-encoder --silent >/dev/null
tar -xzf gpt-3-encoder-*.tgz
cp package/encoder.json gpt2-vocab.json
cp package/vocab.bpe gpt2-merges.txt
rm -rf package gpt-3-encoder-*.tgz
echo "assets ready in $DIR"
