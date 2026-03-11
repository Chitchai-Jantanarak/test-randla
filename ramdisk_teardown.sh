#!/usr/bin/env bash
# ramdisk_teardown.sh — Safely unmount the ramdisk
# Usage: sudo bash ramdisk_teardown.sh [mountpoint]

set -euo pipefail

MOUNTPOINT="${1:-/mnt/ramdisk}"

if mountpoint -q "$MOUNTPOINT"; then
    umount "$MOUNTPOINT"
    echo "[ramdisk] Unmounted $MOUNTPOINT — RAM freed."
else
    echo "[ramdisk] Nothing mounted at $MOUNTPOINT."
fi
