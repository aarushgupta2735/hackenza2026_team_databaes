"""
Audio Language Classifier — CLI entry point.

Usage:
    python -m src.main train [--streaming] [--samples 500]
    python -m src.main predict audio.wav [audio2.mp3 ...]
    python -m src.main predict --dir path/to/folder/
    python -m src.main evaluate
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(
        description="Audio Language Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main train --streaming --samples 200
  python -m src.main predict recording.wav
  python -m src.main predict --dir recordings/
  python -m src.main evaluate
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── train ─────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train the language classifier")
    train_p.add_argument("--raw-dir", type=str, default=None,
                         help="Directory with raw audio (data/raw/<lang>/*.wav)")
    train_p.add_argument("--mdc", action="store_true",
                         help="Download from Mozilla Data Collective (recommended)")
    train_p.add_argument("--streaming", action="store_true",
                         help="Legacy: stream from HuggingFace Common Voice")
    train_p.add_argument("--samples", type=int, default=500,
                         help="Samples per language (default: 500)")

    # ── predict ───────────────────────────────────────────────────
    pred_p = sub.add_parser("predict", help="Predict language of audio files")
    pred_p.add_argument("audio", nargs="*", help="Audio file path(s)")
    pred_p.add_argument("--dir", type=str, default=None,
                        help="Directory of audio files")
    pred_p.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    # ── evaluate ──────────────────────────────────────────────────
    sub.add_parser("evaluate", help="Evaluate on the test set")

    args = parser.parse_args()

    if args.command == "train":
        from src.train import train_model
        train_model(
            raw_dir=args.raw_dir,
            use_mdc=args.mdc,
            use_streaming=args.streaming,
            samples_per_lang=args.samples,
        )

    elif args.command == "predict":
        from src.predict import predict_files
        from src.utils.audio_utils import list_audio_files

        paths = list(args.audio) if args.audio else []
        if args.dir:
            paths.extend(list_audio_files(args.dir))

        if not paths:
            pred_p.print_help()
            sys.exit(1)

        print(f"\nPredicting language for {len(paths)} file(s)...\n")
        results = predict_files(paths)

        if args.json:
            import json
            print(json.dumps(results, indent=2, ensure_ascii=False))

    elif args.command == "evaluate":
        from src.language_classifier import LanguageClassifier
        clf = LanguageClassifier()
        clf.load_dataset()
        clf.extract_features()
        clf.evaluate()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()