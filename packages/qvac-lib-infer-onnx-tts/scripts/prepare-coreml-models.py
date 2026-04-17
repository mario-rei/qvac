#!/usr/bin/env python3
"""Convert ONNX models with external data (.onnx_data) to single-file format.

CoreML EP in ORT <=1.24 cannot load models with external data.
This script inlines all external tensors into the .onnx file itself,
producing a *_coreml.onnx variant that CoreML can load.

Usage:
    python3 convert-to-single-file.py <model_dir> [--suffix _coreml]

Requires: pip install onnx
"""

import argparse
import os
import sys

try:
    import onnx
except ImportError:
    print("Error: 'onnx' package not found. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)


def convert(input_path, output_path):
    model = onnx.load(input_path, load_external_data=True)
    onnx.save_model(model, output_path, save_as_external_data=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="Directory containing .onnx + .onnx_data files")
    parser.add_argument("--suffix", default="_coreml", help="Suffix for output files (default: _coreml)")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"Error: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    converted = 0
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".onnx"):
            continue
        data_file = os.path.join(model_dir, name + "_data")
        if not os.path.exists(data_file):
            continue

        base = name[:-5]  # strip .onnx
        out_name = f"{base}{args.suffix}.onnx"
        out_path = os.path.join(model_dir, out_name)

        if os.path.exists(out_path):
            in_mtime = os.path.getmtime(os.path.join(model_dir, name))
            out_mtime = os.path.getmtime(out_path)
            if out_mtime >= in_mtime:
                print(f"  {out_name} (up to date, skipped)")
                continue

        convert(os.path.join(model_dir, name), out_path)
        converted += 1

    if converted == 0:
        print("Nothing to convert (all up to date or no external-data models found).")
    else:
        print(f"Converted {converted} model(s).")


if __name__ == "__main__":
    main()
