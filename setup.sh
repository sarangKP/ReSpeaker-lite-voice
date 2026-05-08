#!/usr/bin/env bash
# One-shot setup for ReSpeaker Lite voice pipeline.
# Run this once on a new machine, then use: uv run python local.py
set -euo pipefail

echo "==> Installing system dependency: portaudio (required by pyaudio)"
if command -v apt-get &>/dev/null; then
    sudo apt-get install -y portaudio19-dev
elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm portaudio
elif command -v brew &>/dev/null; then
    brew install portaudio
else
    echo "WARNING: Unknown package manager. Install portaudio manually, then re-run this script."
    echo "  Debian/Ubuntu/Pi OS: sudo apt install portaudio19-dev"
    echo "  Arch:                sudo pacman -S portaudio"
    echo "  macOS:               brew install portaudio"
    exit 1
fi

echo ""
echo "==> Syncing Python dependencies via uv..."
uv sync

echo ""
echo "Setup complete."
echo ""
echo "Plug in your ReSpeaker Lite via USB, then run:"
echo "  uv run python local.py"
