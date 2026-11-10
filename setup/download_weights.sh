#!/usr/bin/env bash
# Downloads the model checkpoints into weights/ (they are too large for git).
set -euo pipefail
cd "$(dirname "$0")/.."


BASE="https://github.com/quepanquepun/piano-transcription-tfg/releases/download/v1.0"
# Official checkpoint from the authors: https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases/tag/v0.0.1
M2S_URL="https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases/download/v0.0.1/MIDI2ScoreTF.ckpt"

fetch () {  # url destination
  if [ -f "$2" ]; then echo "✔ $2 already present"; return; fi
  echo "▶ Downloading $(basename "$2")"
  mkdir -p "$(dirname "$2")"
  curl -L --fail -o "$2" "$1"
}

fetch "$BASE/mft_film_maestro_ep30.pt"      weights/mft/mft_film_maestro_ep30.pt
fetch "$BASE/transkun_maestro_ep111.pt"     weights/transkun/transkun_maestro_ep111.pt
fetch "$M2S_URL"      weights/midi2score/MIDI2ScoreTF.ckpt
echo "✔ Weights ready"