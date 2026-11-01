#!/bin/sh
# Real-window screenshots for the README and landing page (Liquid Glass renders only on screen).
#   scripts/shots.sh  → site/assets/screenshot-{chat,thinking,discover,models,server}.jpg
# The app runs in tour mode: sample chats, nothing persisted, real replies generated live, the
# local server started and called, and it captures its own window (no Screen Recording permission
# needed). Keep the window unobstructed. The PNGs have transparent window corners; shot-jpeg.swift
# flattens them onto the page's dark frame colour so no light corners show on the site.
set -eu
cd "$(dirname "$0")/.."
./scripts/bundle.sh >/dev/null 2>&1
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
MLXRUNNER_TOUR="$DIR" "build/MLX Runner.app/Contents/MacOS/MLXRunner" >/dev/null 2>&1
for name in chat thinking discover models server; do
  [ -f "$DIR/$name.png" ] || { echo "missing $name.png" >&2; exit 1; }
  swift scripts/shot-jpeg.swift "$DIR/$name.png" "site/assets/screenshot-$name.jpg" 1568
  echo "site/assets/screenshot-$name.jpg"
done
