#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="$SCRIPT_DIR/../models"
mkdir -p "$MODEL_DIR"

MODEL_FILE="$MODEL_DIR/cc.en.300.bin"
MODEL_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"

if [ -f "$MODEL_FILE" ]; then
    echo "FastText model already exists at $MODEL_FILE, skipping download."
else
    echo "Downloading FastText model..."
    wget -O "$MODEL_FILE.gz" "$MODEL_URL"
    gzip -d -f "$MODEL_FILE.gz"
    [ -f "$MODEL_GZ" ] && rm "$MODEL_GZ"
    echo "FastText model downloaded to $MODEL_FILE"
fi
