#!/usr/bin/env bash
set -euo pipefail

out_dir="dataset/mbpp"
base_url="https://raw.githubusercontent.com/google-research/google-research/master/mbpp"

mkdir -p "$out_dir"

curl -fL "$base_url/README.md" -o "$out_dir/README.md"
curl -fL "$base_url/sanitized-mbpp.json" -o "$out_dir/sanitized-mbpp.json"

echo "Downloaded MBPP files to $out_dir"