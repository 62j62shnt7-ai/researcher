#!/bin/bash
# Double-click to start Researcher. A Terminal window opens and shows progress;
# you don't have to type anything. Close the window (or press Ctrl-C) to stop.

cd "$(dirname "$0")" || exit 1

# Make sure newly installed Python (python.org / Homebrew) is on PATH —
# Finder-launched processes get a minimal PATH.
for d in /usr/local/bin /opt/homebrew/bin \
         /Library/Frameworks/Python.framework/Versions/3.14/bin \
         /Library/Frameworks/Python.framework/Versions/3.13/bin \
         /Library/Frameworks/Python.framework/Versions/3.12/bin \
         /Library/Frameworks/Python.framework/Versions/3.11/bin \
         /Library/Frameworks/Python.framework/Versions/3.10/bin; do
    [ -d "$d" ] && PATH="$d:$PATH"
done
export PATH

# Clear the "downloaded from another machine" flag so macOS doesn't block us.
xattr -dr com.apple.quarantine . 2>/dev/null
chmod +x start_mac.sh 2>/dev/null

echo "Starting Researcher…"
echo "(Your browser will open automatically. Keep this window open while you use the app.)"
echo
exec bash ./start_mac.sh
