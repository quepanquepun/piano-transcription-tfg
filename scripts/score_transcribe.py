"""
Performance MIDI -> MusicXML score with MIDI2ScoreTransformer (Beyer & Dai, ISMIR 2024).

The official repository only provides batch evaluation over the ASAP dataset. This script adds
single-file inference: it tokenizes one MIDI file, decodes it autoregressively in overlapping
chunks (so pieces of any length fit the model's context) and writes a MusicXML score.

Runs inside the `tfg-m2s` conda environment. Normally you call it through ../transcribe.py.

    python score_transcribe.py --input piece.mid --output piece.musicxml --checkpoint ../weights/midi2score/MIDI2ScoreTF.ckpt
"""

import argparse
import os
import sys
import warnings

import torch

CHUNK = 512    # notes per chunk (same length used during training)
OVERLAP = 64   # notes of context carried over from the previous chunk


def import_m2s(m2s_dir: str):
    m2s_dir = os.path.abspath(m2s_dir)
    sys.path.insert(0, m2s_dir)
    sys.path.insert(0, os.path.join(m2s_dir, "models"))
    from models.roformer import Roformer
    from tokenizer import MultistreamTokenizer
    from score_utils import postprocess_score
    return Roformer, MultistreamTokenizer, postprocess_score


def load_model(Roformer, ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ck["hyper_parameters"]  # exact architecture used in training
    model = Roformer(hp["enc_configuration"], hp["dec_configuration"], hyperparameters=hp["hyperparameters"])
    model.load_state_dict(ck["state_dict"], strict=True)  # strict: fail loudly on any mismatch
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[score] model ready on {device} ({n_params / 1e6:.1f}M parameters)")
    return model


@torch.no_grad()
def generate_tokens(model, x: dict, device: str, chunk: int = CHUNK, overlap: int = OVERLAP) -> dict:
    """Autoregressive decoding in overlapping chunks. x: tokenized MIDI streams without batch dim."""
    if chunk <= overlap:
        raise ValueError("chunk must be larger than overlap")

    x = {k: v.unsqueeze(0).to(device) for k, v in x.items()}
    n_notes = x["pitch"].shape[1]
    y_full = None

    for i in range(0, max(n_notes - overlap, 1), chunk - overlap):
        print(f"[score] decoding... {min(100, int(100 * i / n_notes)):3d}%", end="\r", flush=True)
        x_chunk = {k: v[:, i:i + chunk] for k, v in x.items()}

        if y_full is None or overlap == 0:
            y_hat = model.generate(x=x_chunk, top_k=1, max_length=chunk, kv_cache=True)
        else:
            # Prime the decoder with the last `overlap` predicted notes for continuity
            y_prev = {k: (v[:, -overlap:] if k != "pad" else v[:, -overlap:, 0]) for k, v in y_full.items()}
            with torch.autocast(device_type=device):
                y_hat = model.generate(x=x_chunk, y=y_prev, top_k=1, max_length=chunk, kv_cache=True)
            y_hat = {k: v[:, overlap:] for k, v in y_hat.items()}  # drop the primed part

        y_full = y_hat if y_full is None else {k: torch.cat([y_full[k], y_hat[k]], dim=1) for k in y_full}

    print("[score] decoding... 100%")
    return {k: v[0].cpu() for k, v in y_full.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDI2ScoreTransformer: performance MIDI -> MusicXML")
    parser.add_argument("--input", required=True, help="input performance MIDI")
    parser.add_argument("--output", help="output .musicxml (default: same name as input)")
    parser.add_argument("--checkpoint", required=True, help="path to MIDI2ScoreTF.ckpt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--m2s-dir", default=os.path.join(os.path.dirname(__file__), "..", "third_party",
                                                          "MIDI2ScoreTransformer", "midi2scoretransformer"),
                        help="path to the official midi2scoretransformer package folder")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[score] input not found: {args.input}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[score] checkpoint not found: {args.checkpoint} (run setup/download_weights.sh)")
    output = args.output or os.path.splitext(args.input)[0] + ".musicxml"

    Roformer, Tokenizer, postprocess_score = import_m2s(args.m2s_dir)
    model = load_model(Roformer, args.checkpoint, args.device)

    x = Tokenizer.tokenize_midi(args.input)
    print(f"[score] {x['pitch'].shape[0]} notes in {os.path.basename(args.input)}")
    y = generate_tokens(model, x, args.device, args.chunk, args.overlap)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # music21 is very verbose
        score = postprocess_score(Tokenizer.detokenize_mxl(y), inPlace=True)
        score.write("musicxml", fp=output)
    print(f"[score] score -> {output}  (open it with MuseScore)")


if __name__ == "__main__":
    main()
