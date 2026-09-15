#!/bin/bash
# Re-encode legacy mpeg4 ("mp4v") videos to H.264 so VS Code/Chromium can play them.
# New videos are already H.264 (see zerofact/video.py); this fixes files written
# before that change. Safe and idempotent: converts to a temp file, replaces the
# original only on ffmpeg success, and skips files that are already H.264.
#
# Usage: bash scripts/reencode_mp4s_h264.sh [dir]   (default: outputs/)
set -u
root="${1:-$(dirname "$0")/../outputs}"
converted=0; skipped=0; failed=0
while IFS= read -r f; do
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f")
  if [ "$codec" != "mpeg4" ]; then
    skipped=$((skipped+1)); continue
  fi
  tmp="${f%.mp4}.h264tmp.mp4"
  if ffmpeg -nostdin -y -loglevel error -i "$f" \
      -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' \
      -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
      -movflags +faststart -an "$tmp"; then
    mv "$tmp" "$f"; converted=$((converted+1))
  else
    rm -f "$tmp"; failed=$((failed+1)); echo "FAILED: $f"
  fi
done < <(find "$root" -name '*.mp4' -not -name '*.h264tmp.mp4')
echo "converted=$converted skipped=$skipped failed=$failed"
