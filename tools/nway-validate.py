#!/usr/bin/env python3
"""Validation script for the N-Way Weighted GGUF Repacker.

Compares the tensor data from an original monolithic GGUF against the
repacked N-Way chunk files + manifest to verify bitwise correctness.

Usage:
    python tools/nway-validate.py \\
        --original model.gguf \\
        --nway-dir ./nway-out

This reads each routed expert tensor from the original GGUF, reconstructs
it from the chunk fragments described in model-manifest.json, and verifies
byte-for-byte equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing gguf-py as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))

import numpy as np

from gguf.gguf_reader import GGUFReader  # type: ignore[import-untyped]

PAGE_SIZE = 16384  # 16 KB

ROUTED_FAMILIES = frozenset({
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps",
})


def _is_routed_tensor(name: str) -> bool:
    for family in ROUTED_FAMILIES:
        if family in name:
            return True
    return False


def validate(original_path: str, nway_dir: str) -> bool:
    nway = Path(nway_dir)
    manifest_path = nway / "model-manifest.json"

    print(f"[validate] Loading original GGUF: {original_path}")
    reader = GGUFReader(original_path)

    print(f"[validate] Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Open chunk files.
    chunk_files_names = manifest.get("chunk_files", [])
    chunk_files = []
    for name in chunk_files_names:
        path = nway / name
        chunk_files.append(open(path, "rb"))
    print(f"[validate] Opened {len(chunk_files)} chunk files")

    # Build a lookup from tensor name to manifest entry.
    entry_map = {}
    for entry in manifest.get("entries", []):
        entry_map[entry["tensor_name"]] = entry

    # Validate each routed tensor.
    n_checked = 0
    n_passed = 0
    n_failed = 0
    total_bytes = 0

    for tensor in reader.tensors:
        tensor_name = tensor.name
        if isinstance(tensor_name, bytes):
            tensor_name = tensor_name.decode("utf-8")

        if not _is_routed_tensor(tensor_name):
            continue

        if tensor_name not in entry_map:
            print(f"  SKIP {tensor_name} (not in manifest)")
            continue

        n_checked += 1
        entry = entry_map[tensor_name]
        fragments = entry.get("fragments", [])

        # Read original tensor data.
        original_data = tensor.data.tobytes()
        original_size = len(original_data)
        aligned_size = entry.get("aligned_byte_length", original_size)

        # Reconstruct from fragments.
        reconstructed = bytearray(aligned_size)
        cursor = 0
        for frag in fragments:
            file_id = frag["file_id"]
            offset = frag["offset"]
            size = frag["size"]

            chunk_files[file_id].seek(offset)
            frag_data = chunk_files[file_id].read(size)

            if len(frag_data) != size:
                print(f"  FAIL {tensor_name}: fragment read returned {len(frag_data)} bytes, expected {size}")
                n_failed += 1
                continue

            reconstructed[cursor:cursor + size] = frag_data
            cursor += size

        # Compare only the non-padding portion.
        reconstructed_trimmed = bytes(reconstructed[:original_size])

        if reconstructed_trimmed == original_data:
            n_passed += 1
            total_bytes += original_size
            print(f"  PASS {tensor_name} ({original_size:,} bytes, {len(fragments)} fragments)")
        else:
            n_failed += 1
            # Find the first mismatch position.
            for i in range(min(len(original_data), len(reconstructed_trimmed))):
                if original_data[i] != reconstructed_trimmed[i]:
                    print(f"  FAIL {tensor_name}: first mismatch at byte {i} "
                          f"(original=0x{original_data[i]:02x}, reconstructed=0x{reconstructed_trimmed[i]:02x})")
                    break
            else:
                print(f"  FAIL {tensor_name}: length mismatch "
                      f"(original={len(original_data)}, reconstructed={len(reconstructed_trimmed)})")

    # Close chunk files.
    for f in chunk_files:
        f.close()

    # Summary.
    print()
    print(f"[validate] Results: {n_checked} tensors checked, "
          f"{n_passed} passed, {n_failed} failed")
    print(f"[validate] Total validated bytes: {total_bytes / (1024**3):.3f} GiB")

    if n_failed > 0:
        print("[validate] VALIDATION FAILED")
        return False
    elif n_checked == 0:
        print("[validate] WARNING: No routed tensors found to validate")
        return True
    else:
        print("[validate] VALIDATION PASSED - all tensors match bitwise")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate N-Way GGUF repacking against original monolithic GGUF",
    )
    parser.add_argument(
        "--original", "-i",
        required=True,
        help="Path to the original monolithic GGUF file",
    )
    parser.add_argument(
        "--nway-dir", "-d",
        required=True,
        help="Directory containing the N-Way chunks and manifest",
    )

    args = parser.parse_args()
    success = validate(args.original, args.nway_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
