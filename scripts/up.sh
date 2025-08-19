#!/bin/bash
set -e

cd "$(dirname "$0")/.."

bash ./scripts/fasttext_load.sh
uvicorn app.main:app --reload
