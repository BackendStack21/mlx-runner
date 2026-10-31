#!/bin/sh
# Real-window screenshots for the README and landing page (Liquid Glass renders only on screen).
#   scripts/shots.sh  → site/assets/screenshot-{chat,discover,models,server}.jpg
# The app runs in tour mode: sample chats, nothing persisted, a real reply generated live, and it
# captures its own window (no Screen Recording permission needed). Keep the window unobstructed.
set -eu
cd "$(dirname "$0")/.."
./scripts/bundle.sh >/dev/null 2>&1
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
MLXRUNNER_TOUR="$DIR" "build/MLX Runner.app/Contents/MacOS/MLXRunner" >/dev/null 2>&1
for name in chat discover models server; do
  [ -f "$DIR/$name.png" ] || { echo "missing $name.png" >&2; exit 1; }
  sips -s format jpeg -s formatOptions 85 --resampleWidth 1568 "$DIR/$name.png" --out "site/assets/screenshot-$name.jpg" >/dev/null
  echo "site/assets/screenshot-$name.jpg"
done
