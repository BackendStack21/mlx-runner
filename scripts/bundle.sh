#!/bin/sh
# Builds MLX Runner.app from the Swift package (release, arm64) and ad-hoc signs it.
set -e
cd "$(dirname "$0")/.."
swift build -c release --product MLXRunner
APP="build/MLX Runner.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build/release/MLXRunner "$APP/Contents/MacOS/MLXRunner"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>MLX Runner</string>
  <key>CFBundleDisplayName</key><string>MLX Runner</string>
  <key>CFBundleIdentifier</key><string>de.21no.mlxrunner</string>
  <key>CFBundleExecutable</key><string>MLXRunner</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST
codesign --force --sign - "$APP" >/dev/null
echo "Built $APP"
