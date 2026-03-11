#!/usr/bin/env bash
# ramdisk_setup.sh — Mount a tmpfs ramdisk for large point cloud processing
# Usage: sudo bash ramdisk_setup.sh [size] [mountpoint]
#   size       e.g. 200G (default: 200G)
#   mountpoint e.g. /mnt/ramdisk (default: /mnt/ramdisk)

set -euo pipefail

SIZE="${1:-200G}"
MOUNTPOINT="${2:-/mnt/ramdisk}"

# ── Check available RAM ───────────────────────────────────────────────────────
AVAIL_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
SIZE_GB=$(echo "$SIZE" | sed 's/G//')

echo "[ramdisk] Available RAM: ${AVAIL_GB}GB  |  Requested: ${SIZE_GB}GB"
if (( AVAIL_GB < SIZE_GB )); then
    echo "[ramdisk] WARNING: Not enough free RAM. Reduce SIZE or free memory first."
    exit 1
fi

# ── Create and mount ──────────────────────────────────────────────────────────
mkdir -p "$MOUNTPOINT"/{input,output,tmp}

if mountpoint -q "$MOUNTPOINT"; then
    echo "[ramdisk] Already mounted at $MOUNTPOINT — skipping."
else
    mount -t tmpfs -o "size=${SIZE},mode=1777" tmpfs "$MOUNTPOINT"
    echo "[ramdisk] Mounted ${SIZE} tmpfs at $MOUNTPOINT"
fi

echo "[ramdisk] Subdirs: input/  output/  tmp/"
