#!/bin/bash
# Convert the downloaded ImageNet-1k parquet shards into ImageFolder layout.
#
# Runs on the head node by default: the many-core GPU partitions on this cluster
# repeatedly failed to provision (Code:InsufficientInstanceCapacity), and this
# stage is dominated by Lustre write/metadata cost rather than CPU, so 12 workers
# on the head node is a reasonable trade. The converter skips files it has
# already written, so re-running after an interruption is safe and cheap.
#
#   nohup bash scripts/imagenet_convert.sh > logs/convert.log 2>&1 &
set -euo pipefail

# Paths (ENV_PREFIX, PARQUET_DIR, DATA_ROOT) come from .env at the repo root,
# never from here. `set -u` above means an entry missing from .env aborts the
# script by name rather than turning into an empty path.
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
. "$REPO_DIR/.env"

# The conversion's destination is DATA_ROOT: the directory the training code is
# later pointed at with --data-root.
IMAGEFOLDER_DIR=${IMAGEFOLDER_DIR:-$DATA_ROOT}
WORKERS=${WORKERS:-12}

export PATH="$ENV_PREFIX/bin:$PATH"
mkdir -p "$IMAGEFOLDER_DIR"

# Validation first: it is 1/20th the size, so any schema or disk problem shows up
# in a couple of minutes rather than after an hour of train conversion.
echo "=== $(date) converting validation ==="
time python "$REPO_DIR/scripts/prepare_imagenet.py" \
    --parquet-dir "$PARQUET_DIR/data" \
    --out-dir "$IMAGEFOLDER_DIR" \
    --split validation --workers "$WORKERS"

echo "=== $(date) converting train ==="
time python "$REPO_DIR/scripts/prepare_imagenet.py" \
    --parquet-dir "$PARQUET_DIR/data" \
    --out-dir "$IMAGEFOLDER_DIR" \
    --split train --workers "$WORKERS"

echo "=== $(date) verifying ==="
# ImageFolder derives class_to_idx from whichever directories exist, so a partial
# conversion would silently shift every label. Fail loudly instead.
n_train=$(ls "$IMAGEFOLDER_DIR/train" | wc -l)
n_val=$(ls "$IMAGEFOLDER_DIR/val" | wc -l)
echo "train classes: $n_train / 1000"
echo "val classes:   $n_val / 1000"
if [ "$n_train" -ne 1000 ] || [ "$n_val" -ne 1000 ]; then
    echo "ERROR: incomplete conversion -- re-run this script (it resumes)."
    exit 1
fi

echo "train images: $(find "$IMAGEFOLDER_DIR/train" -type f | wc -l)  (expect ~1281167)"
echo "val images:   $(find "$IMAGEFOLDER_DIR/val" -type f | wc -l)  (expect 50000)"
du -sh "$IMAGEFOLDER_DIR"
echo "=== CONVERT COMPLETE ==="
