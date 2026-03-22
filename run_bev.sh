#!/usr/bin/env bash
# BEV inference + apply labels + building counting
# All outputs go to ramdisk for speed
set -euo pipefail

# ---------- Configuration ----------
INPUT="${1:-$HOME/ramdisk/output_half.las}"
OUTDIR="$HOME/ramdisk"
GPU_ID="${2:-1}"
CPUS="${3:-0-95}"
CHUNK_SIZE="${4:-25000000}"

LABELS="$OUTDIR/bev_labels.npy"
CLASSIFIED="$OUTDIR/bev_classified.las"
BUILDINGS_JSON="$OUTDIR/bev_buildings.json"
CONFIG="randlanet_toronto3d_bev_config.yml"

echo "=== BEV Pipeline ==="
echo "  Input:      $INPUT"
echo "  GPU:        $GPU_ID"
echo "  CPUs:       $CPUS"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Labels:     $LABELS"
echo "  Output LAS: $CLASSIFIED"
echo "  Buildings:  $BUILDINGS_JSON"
echo ""

# ---------- Step 1: Inference ----------
echo "=== Step 1: Inference ==="
CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c $CPUS python main.py \
    --input "$INPUT" \
    --labels "$LABELS" \
    --config "$CONFIG" \
    --bev \
    --device cuda \
    --chunk-size "$CHUNK_SIZE"

echo ""
echo "Labels saved: $(ls -lh "$LABELS" | awk '{print $5}')"
echo ""

# ---------- Step 2: Apply labels + count buildings ----------
echo "=== Step 2: Apply Labels + Count Buildings ==="
taskset -c $CPUS python apply.py \
    --input "$INPUT" \
    --labels "$LABELS" \
    --output "$CLASSIFIED" \
    --buildings-json "$BUILDINGS_JSON" \
    --bev \
    --chunk-size 5000000

echo ""
echo "=== Done ==="
echo "  Classified LAS: $(ls -lh "$CLASSIFIED" | awk '{print $5}') → $CLASSIFIED"
echo "  Buildings JSON:  $BUILDINGS_JSON"
echo "  Labels:          $(ls -lh "$LABELS" | awk '{print $5}') → $LABELS"
