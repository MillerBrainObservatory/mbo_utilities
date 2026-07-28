# Installing on macOS

Quick setup for the Miller Brain Studio viewer on a Mac. Run everything in **Terminal**.

Requires [uv](https://docs.astral.sh/uv/). If `uv` isn't installed:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. If you use Anaconda, deactivate it
```
conda deactivate
```
Skip if your prompt doesn't show `(base)`. Re-run in each new Terminal window.

## 2. Remove any previous install (skip if first time)
```
rm -rf ~/projects/mbo_utilities ~/mbo
rm -rf ~/Applications/"Miller Brain Studio.app" ~/Desktop/"Miller Brain Studio.app"
```

## 3. Install from GitHub (master)
```
uv venv ~/projects/mbo_utilities --python 3.12.9
cd ~/projects/mbo_utilities
export GIT_LFS_SKIP_SMUDGE=1
uv pip install --python ./bin/python "git+https://github.com/MillerBrainObservatory/mbo_utilities"
```

## 4. Launch
```
uv run mbo
```
Then use **Open File(s)** or **Select Folder** to load your data.

## 5. (Optional) Make a Desktop app
Paste this whole block to create a double-clickable **Miller Brain Studio** icon:
```
APP="$HOME/Applications/Miller Brain Studio.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$HOME/.mbo"

cat > "$APP/Contents/MacOS/Miller Brain Studio" << 'EOF'
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
MBO="$HOME/projects/mbo_utilities/bin/mbo"
# exec so Python is the app's main process; force arm64 so a Finder launch
# doesn't start under x86_64/Rosetta (which crashes wgpu before the window).
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
  exec /usr/bin/arch -arm64 "$MBO"
else
  exec "$MBO"
fi
EOF
chmod +x "$APP/Contents/MacOS/Miller Brain Studio"

cat > "$APP/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>Miller Brain Studio</string>
    <key>CFBundleIdentifier</key><string>com.millerbrainobservatory.mbo-utilities</string>
    <key>CFBundleName</key><string>Miller Brain Studio</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSUIElement</key><false/>
</dict>
</plist>
EOF

curl -LsSf "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/mbo_utilities/assets/app_settings/icon.png" -o "$APP/Contents/Resources/icon.png" 2>/dev/null || true

ln -sf "$APP" "$HOME/Desktop/Miller Brain Studio.app"
echo "Created: $APP  (+ Desktop alias)"
```
First double-click may be blocked by macOS — allow it once via **System Settings → Privacy & Security → Open Anyway**.

## Updating later
```
cd ~/projects/mbo_utilities
export GIT_LFS_SKIP_SMUDGE=1
uv pip install --python ./bin/python --reinstall-package mbo_utilities "git+https://github.com/MillerBrainObservatory/mbo_utilities"
```
