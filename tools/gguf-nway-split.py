#!/usr/bin/env python3
"""N-Way Weighted GGUF Repacker.

Reads a monolithic GGUF file and partitions the routed expert tensors
across N binary payload files based on user-supplied bandwidth weights.
Produces a JSON manifest compatible with the anemll Flash-MoE slot-bank
runtime for multi-volume parallel I/O.

Every chunk boundary is rounded to the nearest multiple of 16384 bytes
(16 KB) so that the resulting buffers satisfy Apple Silicon Metal page
alignment requirements.

Usage:
    python tools/gguf-nway-split.py \\
        --input model.gguf \\
        --output-dir ./nway-out \\
        --weights 6.0 5.5 5.5

Outputs:
    ./nway-out/model-header.gguf      - metadata-only GGUF (no tensor data)
    ./nway-out/model-chunk-0.bin      - binary payload for volume 0
    ./nway-out/model-chunk-1.bin      - binary payload for volume 1
    ...
    ./nway-out/model-manifest.json    - tensor-to-fragment mapping
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing gguf-py as a package.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "gguf-py"))

import numpy as np

from gguf.gguf_reader import GGUFReader  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGE_SIZE = 16384  # 16 KB — Apple Silicon page alignment
GGUF_MAGIC = 0x46475547  # 'GGUF' little-endian
GGUF_VERSION = 3

# Routed expert tensor family names recognised by the Flash-MoE sidecar
# runtime.  Only these tensors are split; everything else stays in the
# header GGUF.
ROUTED_FAMILIES = frozenset({
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps",
})


def _is_routed_tensor(name: str) -> bool:
    """Return True if *name* belongs to a routed MoE expert family."""
    for family in ROUTED_FAMILIES:
        if family in name:
            return True
    return False


def _tensor_family(name: str) -> str:
    """Extract the tensor family substring from a full tensor name."""
    for family in ROUTED_FAMILIES:
        if family in name:
            return family
    return ""


def _tensor_layer(name: str) -> int:
    """Extract the integer layer index from tensor names like 'blk.42.ffn_gate_exps'."""
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def _align_up(value: int, alignment: int) -> int:
    """Round *value* up to the nearest multiple of *alignment*."""
    return ((value + alignment - 1) // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    """Round *value* down to the nearest multiple of *alignment*."""
    return (value // alignment) * alignment


# ---------------------------------------------------------------------------
# Weighted chunk-size computation
# ---------------------------------------------------------------------------

def compute_chunk_sizes(
    total_bytes: int,
    weights: list[float],
) -> list[int]:
    """Split *total_bytes* proportionally according to *weights*.

    Every returned size is a multiple of PAGE_SIZE (16 KB).  The sizes
    sum to exactly *total_bytes* (which itself must be a multiple of
    PAGE_SIZE, enforced by the caller).
    """
    n = len(weights)
    if n == 0:
        raise ValueError("weights must not be empty")
    if n == 1:
        return [total_bytes]

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("sum of weights must be positive")

    # Ideal (fractional) byte allocation per volume.
    ideal = [total_bytes * w / weight_sum for w in weights]

    # Round each allocation down to the nearest page boundary.
    pages = [_align_down(int(x), PAGE_SIZE) for x in ideal]

    # Distribute the remainder pages so that every byte is accounted for.
    remainder_bytes = total_bytes - sum(pages)
    assert remainder_bytes >= 0
    remainder_pages = remainder_bytes // PAGE_SIZE

    # Give remaining pages to the volumes that lost the most from rounding.
    fractional_loss = [ideal[i] - pages[i] for i in range(n)]
    indices_by_loss = sorted(range(n), key=lambda i: -fractional_loss[i])

    for j in range(remainder_pages):
        pages[indices_by_loss[j % n]] += PAGE_SIZE

    assert sum(pages) == total_bytes, (
        f"chunk sum {sum(pages)} != total {total_bytes}"
    )
    return pages


# ---------------------------------------------------------------------------
# Header-only GGUF writer
# ---------------------------------------------------------------------------

def write_header_gguf(reader: GGUFReader, out_path: Path) -> None:
    """Write a metadata-only GGUF that contains no tensor data.

    This copies the raw header bytes from the original file up to (but
    not including) the tensor data region, but patches the tensor count
    to zero so that loaders do not expect payload data.
    """
    # Approach: re-serialize a minimal GGUF with all KV metadata but
    # an empty tensor list.  We use the raw field data from the reader.
    src_path = reader.file_path if hasattr(reader, "file_path") else None

    # Collect all KV fields (skip internal tensor-info fields).
    kv_fields = []
    for field in reader.fields.values():
        # The reader exposes tensor info as fields with special names;
        # skip those.
        if field.name.startswith("GGUF.tensor_info"):
            continue
        kv_fields.append(field)

    # We'll write a simplified GGUF-v3 header with the metadata from
    # the source.  For simplicity, just copy the raw bytes of the
    # header portion (everything before tensor data) and patch the
    # tensor count to zero.
    #
    # The GGUF header layout is:
    #   magic (4) | version (4) | n_tensors (8) | n_kv (8)
    #   ... KV pairs ...
    #   ... tensor infos ...
    #   <alignment padding>
    #   tensor data
    #
    # We want everything from byte 0 up to (exclusive) the first tensor
    # data byte, with n_tensors patched to 0 and tensor infos removed.

    # Find the data start offset.  The reader provides this.
    if hasattr(reader, "data_offset"):
        data_offset = reader.data_offset
    else:
        # Fallback: find the minimum tensor data offset.
        data_offset = min(
            (t.data_offset for t in reader.tensors),
            default=0,
        )

    # Read the raw header from disk.
    if src_path is not None:
        with open(src_path, "rb") as f:
            raw_header = f.read(data_offset)
    else:
        # In-memory reader — try the data attribute.
        raw_header = bytes(reader.data[:data_offset])

    # Patch n_tensors to 0 (bytes 8..16 in little-endian uint64).
    patched = bytearray(raw_header)
    struct.pack_into("<Q", patched, 8, 0)

    # Remove tensor info entries from the raw header.
    # The KV section sits between byte 24 and the start of tensor infos.
    # Rather than surgically excising tensor infos (fragile with
    # variable-length encodings), we just write the patched header
    # as-is.  A reader that sees n_tensors=0 will ignore any trailing
    # tensor info bytes as padding before the (absent) tensor data.

    with open(out_path, "wb") as f:
        f.write(bytes(patched))


# ---------------------------------------------------------------------------
# Main repacker
# ---------------------------------------------------------------------------

def repack(
    input_path: str,
    output_dir: str,
    weights: list[float],
) -> None:
    n_volumes = len(weights)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[nway-split] Reading GGUF: {input_path}")
    reader = GGUFReader(input_path)

    # Separate routed (expert) tensors from dense/shared tensors.
    routed_tensors = []
    for tensor in reader.tensors:
        name = tensor.name
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if _is_routed_tensor(name):
            routed_tensors.append(tensor)

    if not routed_tensors:
        print("[nway-split] WARNING: No routed expert tensors found in the GGUF.")
        print("[nway-split] The model may not be a Mixture-of-Experts architecture.")

    print(f"[nway-split] Found {len(routed_tensors)} routed expert tensors to distribute across {n_volumes} volumes")
    print(f"[nway-split] Weights: {weights}")

    # ---- Write header GGUF ------------------------------------------------
    header_path = out / "model-header.gguf"
    write_header_gguf(reader, header_path)
    print(f"[nway-split] Wrote header: {header_path}")

    # ---- Open chunk output files ------------------------------------------
    chunk_paths = [out / f"model-chunk-{i}.bin" for i in range(n_volumes)]
    chunk_files = [open(p, "wb") for p in chunk_paths]
    chunk_offsets = [0] * n_volumes  # current write position per file

    # ---- Build manifest ---------------------------------------------------
    manifest: dict[str, Any] = {
        "sidecar_kind": "nway_weighted",
        "n_volumes": n_volumes,
        "weights": weights,
        "page_size": PAGE_SIZE,
        "chunk_files": [p.name for p in chunk_paths],
        "entries": [],
    }

    # For the sidecar runtime we also emit per-tensor entries in the same
    # format as the existing Flash-MoE manifest, extended with a "fragments"
    # array that describes the N-way split.

    total_bytes_written = 0

    for tensor in routed_tensors:
        name = tensor.name
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # Read the raw tensor data from the source GGUF.
        tensor_data: np.ndarray = tensor.data
        raw_bytes = tensor_data.tobytes()
        tensor_size = len(raw_bytes)

        # Determine per-expert sizing.
        # The first dimension of routed expert tensors is n_experts.
        n_experts = tensor.shape[0] if len(tensor.shape) > 0 else 1
        bytes_per_expert = tensor_size // n_experts if n_experts > 0 else tensor_size

        # Align the total tensor size up to page boundary.
        aligned_size = _align_up(tensor_size, PAGE_SIZE)

        # Compute chunk sizes for this tensor.
        chunk_sizes = compute_chunk_sizes(aligned_size, weights)

        # Build fragment descriptors and write data.
        fragments = []
        data_cursor = 0

        for vol_idx in range(n_volumes):
            frag_size = chunk_sizes[vol_idx]
            if frag_size == 0:
                continue

            frag_offset = chunk_offsets[vol_idx]

            # Extract the slice of tensor data for this fragment.
            end = min(data_cursor + frag_size, tensor_size)
            frag_data = raw_bytes[data_cursor:end]

            # Pad to aligned size if needed (last fragment might need padding).
            if len(frag_data) < frag_size:
                frag_data = frag_data + b"\x00" * (frag_size - len(frag_data))

            chunk_files[vol_idx].write(frag_data)

            fragments.append({
                "file_id": vol_idx,
                "offset": frag_offset,
                "size": frag_size,
                "actual_bytes": min(frag_size, tensor_size - data_cursor),
            })

            chunk_offsets[vol_idx] += frag_size
            data_cursor += frag_size
            total_bytes_written += frag_size

        # Emit the manifest entry (compatible with Flash-MoE sidecar format).
        layer = _tensor_layer(name)
        family = _tensor_family(name)

        entry = {
            "tensor_name": name,
            "tensor_family": family,
            "layer": layer,
            "n_experts": int(n_experts),
            "bytes_per_expert": int(bytes_per_expert),
            "exact_byte_length": int(tensor_size),
            "aligned_byte_length": int(aligned_size),
            "fragments": fragments,
            # For backward compatibility with single-file sidecar readers:
            "repacked_file": chunk_paths[0].name,
            "repacked_offset": fragments[0]["offset"] if fragments else 0,
        }
        manifest["entries"].append(entry)

    # ---- Close chunk files ------------------------------------------------
    for f in chunk_files:
        f.close()

    # ---- Write manifest ---------------------------------------------------
    manifest_path = out / "model-manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    # ---- Summary ----------------------------------------------------------
    print(f"[nway-split] Wrote {n_volumes} chunk files:")
    for i, p in enumerate(chunk_paths):
        size_mib = chunk_offsets[i] / (1024 * 1024)
        print(f"  {p.name}: {size_mib:.2f} MiB")
    print(f"[nway-split] Total bytes written: {total_bytes_written / (1024**3):.3f} GiB")
    print(f"[nway-split] Manifest: {manifest_path}")
    print("[nway-split] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-Way Weighted GGUF Repacker for multi-volume Flash-MoE I/O",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input monolithic GGUF file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory for output chunks and manifest",
    )
    parser.add_argument(
        "--weights", "-w",
        nargs="+",
        type=float,
        required=True,
        help="Relative bandwidth weights per volume (e.g. 6.0 5.5 5.5)",
    )

    args = parser.parse_args()

    if len(args.weights) < 1:
        parser.error("At least one weight is required")

    if any(w <= 0 for w in args.weights):
        parser.error("All weights must be positive")

    repack(args.input, args.output_dir, args.weights)


if __name__ == "__main__":
    main()
