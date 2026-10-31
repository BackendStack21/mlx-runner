#!/bin/sh
# Builds MLX Runner.app (release, arm64), ad-hoc signs and verifies it.
#   scripts/bundle.sh            → build/MLX Runner.app
#   scripts/bundle.sh --release  → also dist/release/: the .dmg, the mlxr CLI tarball and SHA256SUMS
set -eu
cd "$(dirname "$0")/.."
VERSION=$(sed -n 's/.*static let current = "\(.*\)".*/\1/p' Sources/MLXEngine/Version.swift)
[ -n "$VERSION" ] || { echo "cannot read the version from Sources/MLXEngine/Version.swift" >&2; exit 1; }
ARCH=arm64

swift build -c release --arch "$ARCH" --product MLXRunner
swift build -c release --arch "$ARCH" --product mlxr
BIN=$(swift build -c release --arch "$ARCH" --show-bin-path)

mkdir -p build
APP="build/MLX Runner.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/MLXRunner" "$APP/Contents/MacOS/MLXRunner"
# App icon, rendered from code (scripts/make-icon.swift) and cached in build/.
if [ ! -f build/AppIcon.icns ] || [ scripts/make-icon.swift -nt build/AppIcon.icns ]; then
  ICONSET=build/AppIcon.iconset
  rm -rf "$ICONSET" && mkdir -p "$ICONSET"
  swift scripts/make-icon.swift build/icon-1024.png >/dev/null
  for s in 16 32 128 256 512; do
    sips -z $s $s build/icon-1024.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) build/icon-1024.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o build/AppIcon.icns
fi
cp build/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>MLX Runner</string>
  <key>CFBundleDisplayName</key><string>MLX Runner</string>
  <key>CFBundleIdentifier</key><string>de.21no.mlxrunner</string>
  <key>CFBundleExecutable</key><string>MLXRunner</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHumanReadableCopyright</key><string>© 2026 21no.de · MIT License</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST
codesign --force --deep --sign - "$APP" 2>/dev/null
codesign --verify --deep --strict "$APP"   # the in-app updater runs the same check
echo "Built $APP ($VERSION)"

[ "${1:-}" = "--release" ] || exit 0

OUT=dist/release
rm -rf "$OUT" && mkdir -p "$OUT"
DMG="$OUT/MLX-Runner-$VERSION-macos-$ARCH.dmg"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ditto "$APP" "$STAGE/MLX Runner.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "MLX Runner" -srcfolder "$STAGE" -fs HFS+ -format ULFO -ov "$DMG" >/dev/null 2>&1
echo "Built $DMG"

CLI="mlxr-$VERSION-macos-$ARCH"
mkdir -p "$STAGE/$CLI"
cp "$BIN/mlxr" "$STAGE/$CLI/mlxr"
codesign --force --sign - "$STAGE/$CLI/mlxr" 2>/dev/null
cp LICENSE "$STAGE/$CLI/"
tar -czf "$OUT/$CLI.tar.gz" -C "$STAGE" "$CLI"
echo "Built $OUT/$CLI.tar.gz"

(cd "$OUT" && shasum -a 256 MLX-Runner-*.dmg mlxr-*.tar.gz > SHA256SUMS)
cat "$OUT/SHA256SUMS"
