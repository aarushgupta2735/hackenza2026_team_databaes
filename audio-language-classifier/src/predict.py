"""
Inference script — predict the language of audio files.

Usage:
    python -m src.predict path/to/audio.wav
    python -m src.predict audio1.wav audio2.mp3 audio3.flac
    python -m src.predict --dir path/to/folder/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.language_classifier import LanguageClassifier
from src.utils.audio_utils import list_audio_files


def predict_files(paths: list[str], verbose: bool = True) -> list[dict]:
    """Predict the language for a list of audio file paths."""
    clf = LanguageClassifier()

    results = []
    for path in paths:
        try:
            result = clf.predict(path)
            result["file"] = os.path.basename(path)
            results.append(result)

            if verbose:
                conf = result["confidence"] * 100
                print(f"  {result['file']:<40} → {result['language']:<12} "
                      f"({conf:.1f}% confidence)")
        except Exception as e:
            print(f"  ⚠ {os.path.basename(path)}: {e}")
            results.append({"file": os.path.basename(path), "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict the language of audio files."
    )
    parser.add_argument(
        "audio", nargs="*",
        help="Path(s) to audio file(s)",
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Directory of audio files to predict",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--model", type=str, default="best",
        choices=["best", "lgb", "svm", "lr"],
        help="Which model to use (default: best)",
    )

    args = parser.parse_args()

    # Collect file paths
    paths = list(args.audio) if args.audio else []
    if args.dir:
        paths.extend(list_audio_files(args.dir))

    if not paths:
        parser.print_help()
        print("\nError: provide at least one audio file or --dir")
        sys.exit(1)

    print(f"\nPredicting language for {len(paths)} file(s)...\n")
    results = predict_files(paths)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main() 