# Automatic Piano Transcription: from Audio to Musical Score

**Bachelor's Thesis — Universitat Politècnica de València (UPV), 2026**
Jorge Rodríguez González · Supervisor: Alfons Juan Císcar

This repository provides the inference pipeline developed in the thesis *Automatic piano transcription using deep
learning*. Given a solo piano recording, the system produces a **performance MIDI** file and, from it, an editable
**MusicXML score**.

```
audio ──► [ MFT-FiLM | Transkun ] ──► performance MIDI ──► [ MIDI2ScoreTransformer ] ──► MusicXML score
```

The full description of the methodology, training procedure, experimental setup and evaluation is available in the
thesis document: [`thesis`](https://riunet.upv.es/entities/publication/e7c74abd-5e89-473a-97ca-24c88b4b5e04).

## Overview

The pipeline is composed of two stages:

1. **Audio-to-MIDI transcription.** Two state-of-the-art acoustic models were trained from scratch on
   MAESTRO v3.0.0 and evaluated following the standard `mir_eval` note-level protocol:
   - **MFT-FiLM** — convolutional-recurrent network on a CQT representation, with frequency-conditioned
     FiLM layers and multi-task prediction of onsets, frames, offsets and velocities.
   - **Transkun** — non-hierarchical Transformer that scores note intervals directly and decodes them with a
     neural semi-Markov CRF, avoiding frame-level post-processing.
2. **MIDI-to-score conversion.** **MIDI2ScoreTransformer**, an encoder–decoder RoFormer that translates
   tokenized performance MIDI into score tokens (pitch spelling, note values, voices, staff assignment). The
   checkpoint released by the original authors is used.

### Contributions of this work

- Training and evaluation of MFT-FiLM and Transkun on MAESTRO v3.0.0; the resulting checkpoints are released here.
- **Single-file score inference** for MIDI2ScoreTransformer, whose public code only supports batch evaluation on
  the ASAP dataset. Pieces of arbitrary length are processed by chunked autoregressive decoding with overlapping
  context (512-note windows, 64-note overlap).
- **Compatibility port** of MIDI2ScoreTransformer to current library versions (`transformers` ≥ 5, Lightning 2):
  the RoFormer layers were re-implemented as plain PyTorch modules, preserving the original parameter layout.
- Standalone MFT-FiLM inference for arbitrary audio files.
- A unified command-line interface with isolated, reproducible environments for each model.

## Installation

Requirements: Linux or WSL2, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and `git`.
A CUDA-capable GPU is recommended; all stages also run on CPU.

```bash
git clone https://github.com/quepanquepun/piano-transcription-tfg.git
cd piano-transcription-tfg
bash setup/setup.sh              # clones the original repositories at pinned commits, applies patches,
                                 # and creates one conda environment per model
bash setup/download_weights.sh   # downloads the model checkpoints into weights/
```

The three models depend on mutually incompatible versions of Python, PyTorch and `transformers`. Each stage
therefore runs in its own environment, and `transcribe.py` — which uses only the Python standard library —
orchestrates them through `conda run`.

## Usage

```bash
python transcribe.py transkun recording.wav       # audio → MIDI (Transkun)
python transcribe.py mft      recording.wav       # audio → MIDI (MFT-FiLM)
python transcribe.py score    recording.mid       # MIDI  → MusicXML
python transcribe.py full     recording.wav --model transkun   # audio → MIDI → MusicXML
```

| Option | Description |
|---|---|
| `-o, --output` | Output path (default: next to the input file) |
| `--device {auto,cuda,cpu}` | Computation device |
| `--weights PATH` | Use an alternative checkpoint (e.g. the original authors' weights) |
| `--dry-run` | Print the commands without executing them |

Accepted audio formats: WAV, MP3, FLAC. The resulting `.musicxml` files can be opened with any notation software,
e.g. [MuseScore](https://musescore.org).


## Repository structure

```
├── transcribe.py              Command-line interface
├── scripts/
│   ├── mft_transcribe.py      Audio → MIDI with MFT-FiLM
│   └── score_transcribe.py    MIDI → MusicXML with MIDI2ScoreTransformer
├── patches/                   Modifications applied to the original MIDI2ScoreTransformer code
├── envs/                      Conda environment specifications (one per model)
├── setup/                     Installation and weight-download scripts
├── weights/                   Model checkpoints (downloaded; not tracked by git)
└── thesis/                    Thesis document and LaTeX sources
```

## Acknowledgements

This work builds on the following open-source projects. Their code is not redistributed in this repository:
`setup/setup.sh` retrieves it from the original sources, and each project retains its own license.

- **MFT-FiLM** — Qi Wang, Mingkuan Liu y Maoshen Jia., "Multitask frequency-time feature-wise linear modulation for piano transcription with pedal," IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2025. https://github.com/Wang-Qi-WQ/MFT-FiLM
- **Transkun** — Y. Yan and Z. Duan, "Scoring Intervals using Non-Hierarchical Transformer for Automatic Piano
  Transcription," *ISMIR*, 2024; Y. Yan, F. Cwitkowitz and Z. Duan, "Skipping the Frame-Level: Event-Based Piano
  Transcription With Neural Semi-CRFs," *NeurIPS*, 2021. https://github.com/Yujia-Yan/Transkun
- **MIDI2ScoreTransformer** — T. Beyer and A. Dai, "End-to-End Piano Performance-MIDI to Score Conversion with
  Transformers," *ISMIR*, 2024. https://github.com/TimFelixBeyer/MIDI2ScoreTransformer
- **MAESTRO** — C. Hawthorne et al., "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO
  Dataset," *ICLR*, 2019.


## License

The code in this repository is released under the [MIT License](LICENSE). The released checkpoints were trained on
MAESTRO (CC BY-NC-SA 4.0) and are provided for non-commercial research purposes only.