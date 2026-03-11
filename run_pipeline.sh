#!/usr/bin/env bash
# run_pipeline.sh — Full ramdisk pipeline:
#   rclone pull → ramdisk → inference → apply → rclone push
#
# Usage:
#   bash run_pipeline.sh [options]
#
# Options (all can also be set as env vars):
#   --remote     RCLONE_REMOTE   rclone remote name (required, e.g. "myremote")
#   --src        RCLONE_SRC      Remote source path  (default: "input")
#   --dst        RCLONE_DST      Remote output path  (default: "output")
#   --ramdisk    RAMDISK         Ramdisk mountpoint  (default: /mnt/ramdisk)
#   --input      INPUT_FILE      Input file name relative to ramdisk/input/
#   --config     CONFIG          YAML config file    (default: randlanet_toronto3d_config.yml)
#   --device     DEVICE          cpu or cuda         (default: cuda)
#   --chunk      CHUNK_SIZE      Chunk size in pts   (default: 500000)
#   --bev                        Enable BEV features
#   --footprints FOOTPRINTS      Footprint file name (relative to ramdisk/input/)
#   --dem        DEM             DEM file name       (relative to ramdisk/input/)
#   --dsm        DSM             DSM file name       (relative to ramdisk/input/)
#   --no-apply                   Skip apply.py step
#   --keep-ramdisk               Do not unmount ramdisk after run
#
# Example:
#   sudo bash run_pipeline.sh \
#       --remote gdrive \
#       --src "LiDAR/raw" \
#       --dst "LiDAR/classified" \
#       --input scan.laz \
#       --bev \
#       --footprints buildings.geojson \
#       --dem dem.tif

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RCLONE_REMOTE="${RCLONE_REMOTE:-}"
RCLONE_SRC="${RCLONE_SRC:-input}"
RCLONE_DST="${RCLONE_DST:-output}"
RAMDISK="${RAMDISK:-/mnt/ramdisk}"
INPUT_FILE="${INPUT_FILE:-}"
CONFIG="${CONFIG:-randlanet_toronto3d_config.yml}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-500000}"
BEV_FLAG=""
FOOTPRINTS="${FOOTPRINTS:-}"
DEM="${DEM:-}"
DSM="${DSM:-}"
SKIP_APPLY=0
KEEP_RAMDISK=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)     RCLONE_REMOTE="$2"; shift 2 ;;
        --src)        RCLONE_SRC="$2";    shift 2 ;;
        --dst)        RCLONE_DST="$2";    shift 2 ;;
        --ramdisk)    RAMDISK="$2";       shift 2 ;;
        --input)      INPUT_FILE="$2";    shift 2 ;;
        --config)     CONFIG="$2";        shift 2 ;;
        --device)     DEVICE="$2";        shift 2 ;;
        --chunk)      CHUNK_SIZE="$2";    shift 2 ;;
        --bev)        BEV_FLAG="--bev";   shift ;;
        --footprints) FOOTPRINTS="$2";    shift 2 ;;
        --dem)        DEM="$2";           shift 2 ;;
        --dsm)        DSM="$2";           shift 2 ;;
        --no-apply)   SKIP_APPLY=1;       shift ;;
        --keep-ramdisk) KEEP_RAMDISK=1;   shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate required args ────────────────────────────────────────────────────
if [[ -z "$RCLONE_REMOTE" ]]; then
    echo "[pipeline] ERROR: --remote is required (e.g. --remote gdrive)"
    exit 1
fi
if [[ -z "$INPUT_FILE" ]]; then
    echo "[pipeline] ERROR: --input is required (e.g. --input scan.laz)"
    exit 1
fi

RAMDISK_IN="${RAMDISK}/input"
RAMDISK_OUT="${RAMDISK}/output"
RAMDISK_TMP="${RAMDISK}/tmp"

# ── Helper: timed step ────────────────────────────────────────────────────────
step() {
    local label="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "[pipeline] STEP: $label"
    echo "════════════════════════════════════════════════════════"
    local start=$SECONDS
    "$@"
    echo "[pipeline] Done in $(( SECONDS - start ))s"
}

# ── Verify ramdisk is mounted ─────────────────────────────────────────────────
if ! mountpoint -q "$RAMDISK"; then
    echo "[pipeline] Ramdisk not mounted at $RAMDISK."
    echo "           Run first: sudo bash ramdisk_setup.sh"
    exit 1
fi

mkdir -p "$RAMDISK_IN" "$RAMDISK_OUT" "$RAMDISK_TMP"

# ── Derived paths ─────────────────────────────────────────────────────────────
STEM="${INPUT_FILE%.*}"          # e.g. scan.laz → scan
INPUT_PATH="${RAMDISK_IN}/${INPUT_FILE}"
LABELS_PATH="${RAMDISK_OUT}/${STEM}_labels.npy"
COLORED_PATH="${RAMDISK_OUT}/${STEM}_colored.ply"
CLASSIFIED_PATH="${RAMDISK_OUT}/${STEM}_classified.las"
BUILDINGS_JSON="${RAMDISK_OUT}/${STEM}_buildings.json"

# ── Optional feature file args ────────────────────────────────────────────────
FOOTPRINTS_ARG=""
DEM_ARG=""
DSM_ARG=""
[[ -n "$FOOTPRINTS" ]] && FOOTPRINTS_ARG="--footprints ${RAMDISK_IN}/${FOOTPRINTS}"
[[ -n "$DEM" ]]        && DEM_ARG="--dem ${RAMDISK_IN}/${DEM}"
[[ -n "$DSM" ]]        && DSM_ARG="--dsm ${RAMDISK_IN}/${DSM}"

# ── Python runner (uv if available, else plain python) ────────────────────────
PY_RUN="python"
if command -v uv &>/dev/null; then
    PY_RUN="uv run python"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Pull inputs from remote into ramdisk
# ══════════════════════════════════════════════════════════════════════════════
step "rclone pull → ramdisk/input" \
    rclone copy \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --buffer-size 256M \
        "${RCLONE_REMOTE}:${RCLONE_SRC}/" \
        "${RAMDISK_IN}/"

# ── Sanity check ──────────────────────────────────────────────────────────────
if [[ ! -f "$INPUT_PATH" ]]; then
    echo "[pipeline] ERROR: Input file not found after rclone pull: $INPUT_PATH"
    echo "           Files in ramdisk/input/:"
    ls -lh "$RAMDISK_IN/" || true
    exit 1
fi
echo "[pipeline] Input: $(du -sh "$INPUT_PATH" | cut -f1)  $INPUT_PATH"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Run inference (main.py)
# ══════════════════════════════════════════════════════════════════════════════
step "main.py inference" \
    $PY_RUN "${SCRIPT_DIR}/main.py" \
        --input      "$INPUT_PATH" \
        --labels     "$LABELS_PATH" \
        --output     "$COLORED_PATH" \
        --config     "${SCRIPT_DIR}/${CONFIG}" \
        --device     "$DEVICE" \
        --chunk-size "$CHUNK_SIZE" \
        $BEV_FLAG \
        $FOOTPRINTS_ARG \
        $DEM_ARG \
        $DSM_ARG

echo "[pipeline] Labels: $(du -sh "$LABELS_PATH" | cut -f1)  $LABELS_PATH"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Apply labels + building detection (apply.py)
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$SKIP_APPLY" -eq 0 ]]; then
    step "apply.py post-processing" \
        $PY_RUN "${SCRIPT_DIR}/apply.py" \
            --input          "$INPUT_PATH" \
            --labels         "$LABELS_PATH" \
            --output         "$CLASSIFIED_PATH" \
            --buildings-json "$BUILDINGS_JSON" \
            --chunk-size     "$CHUNK_SIZE"

    echo "[pipeline] Classified: $(du -sh "$CLASSIFIED_PATH" | cut -f1)  $CLASSIFIED_PATH"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Push outputs from ramdisk to remote
# ══════════════════════════════════════════════════════════════════════════════
step "rclone push ramdisk/output → remote" \
    rclone copy \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --buffer-size 256M \
        "${RAMDISK_OUT}/" \
        "${RCLONE_REMOTE}:${RCLONE_DST}/"

echo ""
echo "[pipeline] Output files pushed to ${RCLONE_REMOTE}:${RCLONE_DST}/"
echo "[pipeline] Total output size: $(du -sh "$RAMDISK_OUT" | cut -f1)"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Cleanup ramdisk
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$KEEP_RAMDISK" -eq 0 ]]; then
    step "cleanup ramdisk" bash -c "
        rm -rf '${RAMDISK_IN:?}/'* '${RAMDISK_OUT:?}/'* '${RAMDISK_TMP:?}/'*
        echo '[pipeline] Ramdisk contents cleared (still mounted).'
    "
    echo "[pipeline] To unmount ramdisk: sudo bash ${SCRIPT_DIR}/ramdisk_teardown.sh"
fi

echo ""
echo "[pipeline] ✓ Pipeline complete."
