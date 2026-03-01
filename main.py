#!/usr/bin/env python3
"""
main.py — Native / Non-Native Arabic Speaker Classifier
========================================================
Entry point for the hackenza2026_team_databaes project.

Runs inference on the unlabelled test dataset using pre-trained
models (LSTM encoder + SVM RBF) to classify each audio recording
as Native or Non-Native Arabic speaker.

Usage:
  # Default: runs on the provided test dataset
  python main.py

  # Custom input CSV and output directory
  python main.py --audio-csv data/my_test.csv --output-dir results

  # Specify models directory (if different from default)
  python main.py --models-dir src/native_non-native_model/models

  # Force CPU even if GPU is available
  python main.py --device cpu
"""

import argparse
import importlib.util
import os
import sys

# Ensure the project root is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import pipeline from hyphenated directory name using importlib
_spec = importlib.util.spec_from_file_location(
    "pipeline",
    os.path.join(PROJECT_ROOT, "src", "native_non-native_model", "pipeline.py"),
)
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
predict_from_urls = _pipeline.predict_from_urls


def main():
    parser = argparse.ArgumentParser(
        description="Native / Non-Native Arabic Speaker Classifier — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --audio-csv data/custom_test.csv --output-dir my_outputs
  python main.py --device cpu
        """,
    )
    parser.add_argument(
        "--audio-csv",
        default=os.path.join(PROJECT_ROOT, "data", "Nativity Assessmet Audio Dataset(Test Dataset).csv"),
        help="CSV with dp_id and audio_url columns (default: provided test dataset)",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join(PROJECT_ROOT, "src", "native_non-native_model", "models"),
        help="Directory containing saved models (lstm_encoder.pt.zip, svm_model.joblib, scaler.joblib, lstm_config.joblib)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "src", "native_non-native_model", "outputs"),
        help="Directory to save predictions.csv",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.audio_csv):
        print(f"ERROR: Audio CSV not found: {args.audio_csv}")
        sys.exit(1)
    if not os.path.isdir(args.models_dir):
        print(f"ERROR: Models directory not found: {args.models_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Native / Non-Native Arabic Speaker Classifier")
    print("  Team Databaes — Hackenza 2026")
    print("=" * 60)
    print(f"  Audio CSV   : {args.audio_csv}")
    print(f"  Models dir  : {args.models_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 60)

    results = predict_from_urls(
        audio_csv=args.audio_csv,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Print summary table
    valid = results[results["predicted_class"].isin(["Native", "Non-Native"])]
    if len(valid) > 0:
        print("\n" + "=" * 60)
        print("  PREDICTION SUMMARY")
        print("=" * 60)
        native_count = (valid["predicted_class"] == "Native").sum()
        non_native_count = (valid["predicted_class"] == "Non-Native").sum()
        avg_confidence = valid["confidence_score"].mean()
        print(f"  Total predictions : {len(valid)}")
        print(f"  Native            : {native_count}")
        print(f"  Non-Native        : {non_native_count}")
        print(f"  Avg confidence    : {avg_confidence:.2%}")
        print("=" * 60)

    errors = results[~results["predicted_class"].isin(["Native", "Non-Native"])]
    if len(errors) > 0:
        print(f"\n  WARNING: {len(errors)} files could not be classified.")
        print(errors[["dp_id", "predicted_class", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
