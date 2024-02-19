#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MODEL_NAME=${1:-bge-large-zh-v1.5}
MODEL_DIR=$(realpath "$SCRIPT_DIR/../models/$MODEL_NAME")

if [ ! -d "$MODEL_DIR" ]; then
  python3 $SCRIPT_DIR/../models/download-repo.py $MODEL_NAME
fi

if [ ! -d "$MODEL_DIR/ggml-model-q4_1.gguf" ]; then
  $SCRIPT_DIR/../models/run_conversions.sh $MODEL_NAME q4_1
fi

python3 $SCRIPT_DIR/test_hf_tokenizer.py $MODEL_DIR

$SCRIPT_DIR/../build/bin/test_tokenizer -m $MODEL_DIR/ggml-model-q4_1.gguf
