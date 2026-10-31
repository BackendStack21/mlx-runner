#!/bin/sh
# Real-window screenshots for the README and landing page (Liquid Glass renders only on screen).
#   scripts/shots.sh            → site/assets/screenshot-{chat,discover,models,server}.jpg
# Needs Screen Recording permission for your terminal (System Settings → Privacy & Security).
# The app runs in tour mode: sample chats, nothing persisted, a real reply generated live.
set -eu
cd "$(dirname "$0")/.."
./scripts/bundle.sh >/dev/null
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
screencapture -x "$DIR/probe.png" 2>/dev/null || { echo "screencapture failed: grant Screen Recording to your terminal" >&2; exit 1; }
MLXRUNNER_TOUR="$DIR" "build/MLX Runner.app/Contents/MacOS/MLXRunner" >/dev/null 2>&1 &
APP=$!
for name in chat discover models server; do
  i=0
  while [ ! -f "$DIR/ready-$name" ]; do
    i=$((i + 1)); [ $i -lt 1200 ] || { echo "timed out waiting for $name" >&2; kill $APP; exit 1; }
    sleep 0.1
  done
  screencapture -x -o -l "$(cat "$DIR/ready-$name")" "$DIR/$name.png"
  sips -s format jpeg -s formatOptions 85 --resampleWidth 1568 "$DIR/$name.png" --out "site/assets/screenshot-$name.jpg" >/dev/null
  echo "site/assets/screenshot-$name.jpg"
done
wait $APP 2>/dev/null || true
