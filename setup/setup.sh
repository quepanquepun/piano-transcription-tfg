#!/usr/bin/env bash
# One-time setup: fetch the official repositories at the exact commits used in the thesis,
# apply the compatibility patch and create the three conda environments.
set -euo pipefail
cd "$(dirname "$0")/.."

MFT_REPO="https://github.com/Wang-Qi-WQ/MFT-FiLM.git"
MFT_COMMIT="a65a86c84c0f8e84d57d0941f7a12c77285df11a"
M2S_REPO="https://github.com/TimFelixBeyer/MIDI2ScoreTransformer.git"
M2S_COMMIT="115432bda16ca16e0fec2e9465788f2ba369971f"

mkdir -p third_party

if [ ! -d third_party/MFT-FiLM ]; then
  echo "▶ Cloning MFT-FiLM"
  git clone "$MFT_REPO" third_party/MFT-FiLM
  git -C third_party/MFT-FiLM checkout -q "$MFT_COMMIT"
fi

if [ ! -d third_party/MIDI2ScoreTransformer ]; then
  echo "▶ Cloning MIDI2ScoreTransformer"
  git clone "$M2S_REPO" third_party/MIDI2ScoreTransformer
  git -C third_party/MIDI2ScoreTransformer checkout -q "$M2S_COMMIT"
  echo "▶ Applying compatibility patch (transformers 5.x / lightning 2.x)"
  git -C third_party/MIDI2ScoreTransformer apply "$PWD/patches/midi2scoretransformer.patch"
fi

for env in mft transkun midi2score; do
  name=$(grep '^name:' "envs/$env.yml" | awk '{print $2}')
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "✔ conda env $name already exists"
  else
    echo "▶ Creating conda env $name"
    conda env create -f "envs/$env.yml"
  fi
done

echo
echo "✔ Setup complete. Now download the weights:  bash setup/download_weights.sh"
