#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/ft_simple.yaml}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29517}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi
export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
cd "$SCRIPT_DIR"
exec torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node "$GPUS" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --device-type "$DEVICE_TYPE"
