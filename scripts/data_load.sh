#!/bin/bash
set -e

PROJECT_DIR="$(dirname "$(realpath "$0")")/.."
TARGET_DIR="$PROJECT_DIR/data/raw"
mkdir -p "$TARGET_DIR"

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p "$TARGET_DIR"

cd "$TARGET_DIR"
unzip -qo jigsaw-toxic-comment-classification-challenge.zip

rm -f jigsaw-toxic-comment-classification-challenge.zip

echo "Data loaded to $TARGET_DIR"