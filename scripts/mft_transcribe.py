"""
Audio -> performance MIDI with MFT-FiLM.

Runs inside the `tfg-mft` conda environment, with the official MFT-FiLM repository
on the Python path (third_party/MFT-FiLM). Normally you call it through ../transcribe.py.

    python mft_transcribe.py --input piece.wav --output piece.mid --checkpoint ../weights/mft/model.pt
"""

import argparse
import os
import sys

import pretty_midi

MIDI_A0 = 21  # the model predicts 88 pitch bins (0-87); bin 0 = A0 = MIDI note 21


def import_mft(mft_dir: str):
    """Make the official MFT-FiLM code importable and return the modules we need."""
    sys.path.insert(0, os.path.abspath(mft_dir))
    import constant_macro as cm   # hyper-parameters and thresholds of the original repo
    import checkpoint
    import transcribe
    from net import mt_film
    from utils import metric

    # Checkpoints saved during training pickle the class as `checkpoint.ModelInfo1`;
    # alias it so they can be un-pickled without editing the original repository.
    if not hasattr(checkpoint, "ModelInfo1"):
        checkpoint.ModelInfo1 = checkpoint.ModelInfo
    return cm, transcribe, mt_film, metric


def save_midi(pitches, intervals, velocities, output_path: str) -> None:
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # acoustic grand piano
    for p, (start, end), v in zip(pitches, intervals, velocities):
        piano.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p) + MIDI_A0,
                                            start=float(start), end=float(end)))
    pm.instruments.append(piano)
    pm.write(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MFT-FiLM: audio -> performance MIDI")
    parser.add_argument("--input", required=True, help="input audio (wav/mp3/flac)")
    parser.add_argument("--output", required=True, help="output MIDI file")
    parser.add_argument("--checkpoint", required=True, help="path to the .pt checkpoint")
    parser.add_argument("--onset-th", type=float, help="onset threshold (default: ONS_TH in constant_macro.py)")
    parser.add_argument("--frame-th", type=float, help="frame threshold (default: FRM_TH in constant_macro.py)")
    parser.add_argument("--offset-th", type=float, help="offset threshold (default: OFF_TH in constant_macro.py)")
    parser.add_argument("--mft-dir", default=os.path.join(os.path.dirname(__file__), "..", "third_party", "MFT-FiLM"),
                        help="path to the official MFT-FiLM repository")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[mft] input not found: {args.input}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"[mft] checkpoint not found: {args.checkpoint} (run setup/download_weights.sh)")

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    ckp_dir, ckp_name = os.path.split(os.path.abspath(args.checkpoint))

    cm, transcribe, mt_film, metric = import_mft(args.mft_dir)

    # 1. Model -----------------------------------------------------------------
    if cm.MODEL_NAME == "MT_FiLM":
        arch = mt_film.MT_FiLM()
    elif cm.MODEL_NAME == "TF_FiLMSustainPedal":
        arch = mt_film.MT_FiLMSustainPedal()
    else:
        sys.exit(f"[mft] unknown MODEL_NAME in constant_macro.py: {cm.MODEL_NAME}")

    print(f"[mft] loading {ckp_name}")
    model, _, device, _, _ = transcribe.init_model(arch, ckp_dir + os.sep, ckp_name)
    print(f"[mft] model ready on {device}")

    # 2. Audio -> CQT spectrogram ---------------------------------------------------
    print(f"[mft] computing CQT for {os.path.basename(input_path)}")
    frames, _ = transcribe.convert_to_spectrum(audio_path=input_path, preprocess="cqt",
                                               with_ground_truth=False)
    spectrum = frames["cqt"]
    # Upstream inference path transposes the CQT twice; restore (frames, 352 bins).
    if spectrum.shape[0] == 352:
        spectrum = spectrum.T

    # 3. Network inference -> onset / frame / offset / velocity probabilities ----------
    print("[mft] running inference")
    out = transcribe.piano_transcription(model, spectrum, device, block_stride=0.5,
                                         schedule_idx=cm.SCHEDULE_IDX, evaluate_term="note")

    # 4. Decoding: local-maximum note picking (as in the High-Resolution Piano Transcription paper)
    pitches, intervals, velocities = metric.hpt_local_maximum_algorithm(
        onsets=out["onset_pred"].cpu(),
        frames=out["frame_pred"].cpu(),
        offsets=out["offset_pred"].cpu(),
        velocities=out["velocity_pred"].cpu(),
        onset_threshold=cm.ONS_TH if args.onset_th is None else args.onset_th,
        frame_threshold=cm.FRM_TH if args.frame_th is None else args.frame_th,
        offset_threshold=cm.OFF_TH if args.offset_th is None else args.offset_th,
        decoding_with_offsets=True,
        vel_rescale=cm.VELOCITY_RESCALE,
        scaling=cm.FRAME_LENGTH_MS / 1000 * cm.HOP_LENGTH_RATIO,  # frames -> seconds
    )
    if len(pitches) == 0:
        sys.exit("[mft] no notes detected — is this a piano recording?")

    # 5. Save ------------------------------------------------------------------
    save_midi(pitches, intervals, velocities, output_path)
    print(f"[mft] {len(pitches)} notes -> {output_path}")


if __name__ == "__main__":
    main()
