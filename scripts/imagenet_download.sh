#!/bin/bash
# Download the ImageNet-1k parquet shards (train + validation, ~153GB).
#
# This runs on the head node rather than through Slurm: the cluster has no
# CPU-only partition, and the many-core GPU partitions repeatedly failed to
# provision with Code:InsufficientInstanceCapacity. Downloading is network-bound,
# so the head node's 16 cores are not the limit -- and `hf download` resumes, so
# an interrupted run just needs re-invoking.
#
#   nohup bash scripts/imagenet_download.sh > logs/download.log 2>&1 &
set -euo pipefail

# Paths (ENV_PREFIX, PARQUET_DIR, HF_HUB_CACHE) come from .env at the repo root,
# never from here. `set -u` above means an entry missing from .env aborts the
# script by name rather than turning into an empty path.
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
. "$REPO_DIR/.env"

export PATH="$ENV_PREFIX/bin:$PATH"
# Redirect only the blob cache; the token stays at the default HF_HOME.
export HF_HUB_CACHE

mkdir -p "$PARQUET_DIR" "$HF_HUB_CACHE"

echo "=== $(date) downloading train + validation parquet ==="
# Each glob needs its own --include flag: a bare second pattern is parsed as a
# positional filename and requested verbatim (which 404s). Modest worker count
# because this is a shared login node.
hf download ILSVRC/imagenet-1k \
    --repo-type dataset \
    --include "data/train-*.parquet" \
    --include "data/validation-*.parquet" \
    --local-dir "$PARQUET_DIR" \
    --max-workers 8

n=$(ls "$PARQUET_DIR/data"/train-*.parquet 2>/dev/null | wc -l)
v=$(ls "$PARQUET_DIR/data"/validation-*.parquet 2>/dev/null | wc -l)
echo "=== $(date) done: $n train shards, $v validation shards ==="
du -sh "$PARQUET_DIR"

if [ "$n" -ne 294 ] || [ "$v" -ne 14 ]; then
    echo "ERROR: expected 294 train + 14 validation shards, got $n + $v"
    exit 1
fi
echo "=== DOWNLOAD COMPLETE ==="
