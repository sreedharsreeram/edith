#!/bin/sh
# Regenerates the spoken-question fixtures (macOS only; the WAVs are committed so tests run anywhere).
cd "$(dirname "$0")/audio" || exit 1
while IFS='|' read -r name text; do
  say -o "$name.wav" --data-format=LEI16@16000 "$text"
done <<'LIST'
plant|What kind of plant is this?
calories|How many calories are in this?
translate|Translate this sign into English.
looking|What am I looking at?
LIST
