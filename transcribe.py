#!/usr/bin/env python3
"""
Piano transcription pipeline — Bachelor's Thesis, Jorge Rodríguez González (UPV, 2026).

Each model lives in its own conda environment (their dependencies are incompatible),
so this script only orchestrates: it uses the standard library and calls each stage with
`conda run`. You can run it from any environment.

    python transcribe.py mft      piece.wav            # audio -> MIDI   (MFT-FiLM)
    python transcribe.py transkun piece.wav            # audio -> MIDI   (Transkun)
    python transcribe.py score    piece.mid            # MIDI  -> MusicXML (MIDI2ScoreTransformer)
    python transcribe.py full     piece.wav --model transkun   # audio -> MIDI -> MusicXML
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
WEIGHTS = ROOT / "weights"
THIRD_PARTY = ROOT / "third_party"

ENVS = {  # conda environment per stage (override with env vars if you named them differently)
    "mft": os.environ.get("TFG_ENV_MFT", "tfg-mft"),
    "transkun": os.environ.get("TFG_ENV_TRANSKUN", "tfg-transkun"),
    "score": os.environ.get("TFG_ENV_SCORE", "tfg-m2s"),
}

DEFAULT_WEIGHTS = {
    "mft": WEIGHTS / "mft" / "mft_film_maestro_ep30.pt",          # trained by me on MAESTRO v3 (epoch 30)
    "transkun": WEIGHTS / "transkun" / "transkun_maestro_ep111.pt",  # trained by me on MAESTRO v3 (epoch 111)
    "score": WEIGHTS / "midi2score" / "MIDI2ScoreTF.ckpt",        # official checkpoint (Beyer & Dai)
}
TRANSKUN_CONF = WEIGHTS / "transkun" / "conf.json"  # architecture config saved during training

# Decoding thresholds used in the thesis evaluation for MFT-FiLM (onset, frame, offset).
MFT_THRESHOLDS = (0.4, 0.48, 0.5)


# ---------------------------------------------------------------------------------------
def conda_run(env: str, cmd: list, cwd: Path = None, dry_run: bool = False) -> None:
    full = ["conda", "run", "--no-capture-output", "-n", env] + [str(c) for c in cmd]
    if dry_run:
        print("[dry-run]", f"(cwd={cwd})" if cwd else "", " ".join(full))
        return
    if shutil.which("conda") is None:
        sys.exit("conda not found. Install Miniconda and run setup/setup.sh first.")
    t0 = time.time()
    result = subprocess.run(full, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Stage failed in environment '{env}' (exit code {result.returncode}).")
    print(f"  done in {time.time() - t0:.1f}s\n")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def default_output(inp: Path, suffix: str, tag: str = "") -> Path:
    return inp.with_name(inp.stem + (f"_{tag}" if tag else "") + suffix)


# --- stages ----------------------------------------------------------------------------
def run_mft(audio: Path, out: Path, weights: Path, dry_run: bool) -> None:
    print(f"▶ MFT-FiLM: {audio.name} → {out.name}")
    conda_run(ENVS["mft"], ["python", SCRIPTS / "mft_transcribe.py", "--input", audio, "--output", out,
                            "--checkpoint", weights, "--mft-dir", THIRD_PARTY / "MFT-FiLM"]
              + (["--onset-th", MFT_THRESHOLDS[0], "--frame-th", MFT_THRESHOLDS[1], "--offset-th", MFT_THRESHOLDS[2]]
                 if MFT_THRESHOLDS else []),
              cwd=THIRD_PARTY / "MFT-FiLM", dry_run=dry_run)


def run_transkun(audio: Path, out: Path, weights: Path, device: str, dry_run: bool) -> None:
    print(f"▶ Transkun: {audio.name} → {out.name}")
    cmd = ["transkun", audio, out, "--device", resolve_device(device)]
    if weights.is_file():
        cmd += ["--weight", weights]
        if TRANSKUN_CONF.is_file():
            cmd += ["--conf", TRANSKUN_CONF]
    else:
        print("  (no custom weights found — using the official weights bundled with transkun)")
    conda_run(ENVS["transkun"], cmd, dry_run=dry_run)


def run_score(midi: Path, out: Path, weights: Path, device: str, dry_run: bool) -> None:
    print(f"▶ MIDI2ScoreTransformer: {midi.name} → {out.name}")
    m2s_dir = THIRD_PARTY / "MIDI2ScoreTransformer" / "midi2scoretransformer"
    conda_run(ENVS["score"], ["python", SCRIPTS / "score_transcribe.py", "--input", midi, "--output", out,
                              "--checkpoint", weights, "--device", resolve_device(device),
                              "--m2s-dir", m2s_dir],
              cwd=m2s_dir, dry_run=dry_run)


# --- CLI -------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Piano audio → MIDI → sheet music (Bachelor's Thesis, UPV 2026)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, in_help):
        p.add_argument("input", type=Path, help=in_help)
        p.add_argument("-o", "--output", type=Path, help="output file (default: next to the input)")
        p.add_argument("--weights", type=Path, help="checkpoint to use instead of the default one")
        p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--dry-run", action="store_true", help="print the commands without running them")

    add_common(sub.add_parser("mft", help="audio → MIDI with MFT-FiLM"), "piano recording (wav/mp3/flac)")
    add_common(sub.add_parser("transkun", help="audio → MIDI with Transkun"), "piano recording (wav/mp3/flac)")
    add_common(sub.add_parser("score", help="performance MIDI → MusicXML score"), "performance MIDI file")
    p_full = sub.add_parser("full", help="audio → MIDI → MusicXML in one go")
    add_common(p_full, "piano recording (wav/mp3/flac)")
    p_full.add_argument("--model", choices=["mft", "transkun"], default="transkun", help="audio → MIDI model")

    a = parser.parse_args()
    inp = a.input.resolve()
    if not inp.is_file() and not a.dry_run:
        sys.exit(f"Input file not found: {inp}")

    if a.command in ("mft", "transkun"):
        out = (a.output or default_output(inp, ".mid", a.command)).resolve()
        w = (a.weights or DEFAULT_WEIGHTS[a.command]).resolve()
        if a.command == "mft":
            run_mft(inp, out, w, a.dry_run)
        else:
            run_transkun(inp, out, w, a.device, a.dry_run)

    elif a.command == "score":
        out = (a.output or default_output(inp, ".musicxml")).resolve()
        run_score(inp, out, (a.weights or DEFAULT_WEIGHTS["score"]).resolve(), a.device, a.dry_run)

    else:  # full pipeline
        midi = default_output(inp, ".mid", a.model).resolve()
        xml = (a.output or default_output(inp, ".musicxml", a.model)).resolve()
        w = (a.weights or DEFAULT_WEIGHTS[a.model]).resolve()
        if a.model == "mft":
            run_mft(inp, midi, w, a.dry_run)
        else:
            run_transkun(inp, midi, w, a.device, a.dry_run)
        run_score(midi, xml, DEFAULT_WEIGHTS["score"].resolve(), a.device, a.dry_run)
        print(f"✔ MIDI:  {midi}\n✔ Score: {xml}")


if __name__ == "__main__":
    main()
