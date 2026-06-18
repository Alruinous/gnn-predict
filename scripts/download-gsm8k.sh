#!/usr/bin/env bash
set -euo pipefail

out_dir="dataset/gsm8k"
base_url="https://raw.githubusercontent.com/openai/grade-school-math/master"

mkdir -p "$out_dir/data"

curl -fL "$base_url/README.md" -o "$out_dir/README.md"
curl -fL "$base_url/LICENSE" -o "$out_dir/LICENSE"
curl -fL "$base_url/grade_school_math/data/train.jsonl" -o "$out_dir/data/train.jsonl"
curl -fL "$base_url/grade_school_math/data/test.jsonl" -o "$out_dir/data/test.jsonl"

echo "Downloaded GSM8K files to $out_dir"