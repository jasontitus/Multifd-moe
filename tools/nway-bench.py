#!/usr/bin/env python3
"""N-Way I/O Benchmark — evaluate multi-volume drive throughput and recommend weights.

This tool mirrors the I/O pattern that llama.cpp's Flash-MoE slot-bank runtime
performs during a generation run: sequential pread() calls of page-aligned expert
tensors, fanned out across multiple volumes in parallel.

It reads the manifest produced by gguf-nway-split.py, opens the chunk files,
and simulates expert fetch traffic at realistic sizes.  Reports p50/p99 latency
and throughput per volume, aggregate throughput, and recommends --weights values
that match measured drive performance.

Usage — benchmark existing nway split:
    python tools/nway-bench.py \\
        --manifest ./nway-out/model-manifest.json \\
        --iterations 200

Usage — benchmark raw drive paths (no manifest needed):
    python tools/nway-bench.py \\
        --drives /Volumes/NVMe/bench /Volumes/TB4_1/bench /Volumes/TB4_2/bench \\
        --block-size 2097152 \\
        --iterations 500

The --drives mode creates temporary benchmark files and measures raw sequential
read throughput per drive, which is useful for evaluating a drive setup before
repacking a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGE_SIZE = 16384  # 16 KB — Apple Silicon Metal alignment
DEFAULT_BLOCK_SIZE = 2 * 1024 * 1024  # 2 MiB — typical expert tensor size
DEFAULT_ITERATIONS = 200
DEFAULT_FILE_SIZE = 256 * 1024 * 1024  # 256 MiB bench file per drive
WARMUP_ITERATIONS = 10


@dataclass
class PreadResult:
    """Result of a single pread operation."""
    fd_idx: int
    latency_us: float
    bytes_read: int
    throughput_gbps: float


@dataclass
class VolumeStats:
    """Accumulated statistics for a single volume."""
    path: str
    ops: int = 0
    total_bytes: int = 0
    total_us: float = 0.0
    latencies_us: list[float] = field(default_factory=list)
    throughputs_gbps: list[float] = field(default_factory=list)


def percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a sorted list."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(pct / 100.0 * len(s)), len(s) - 1)
    return s[idx]


def do_pread(fd: int, size: int, offset: int, fd_idx: int) -> PreadResult:
    """Perform a single pread and measure latency."""
    t0 = time.monotonic()
    data = os.pread(fd, size, offset)
    elapsed_us = (time.monotonic() - t0) * 1e6
    n = len(data)
    gbps = n / (elapsed_us * 1000.0) if elapsed_us > 0 else 0.0
    return PreadResult(fd_idx=fd_idx, latency_us=elapsed_us, bytes_read=n, throughput_gbps=gbps)


# ---------------------------------------------------------------------------
# Manifest-based benchmark
# ---------------------------------------------------------------------------

def bench_manifest(manifest_path: str, iterations: int, parallel: bool) -> list[VolumeStats]:
    """Benchmark using an existing nway manifest + chunk files."""
    manifest_dir = Path(manifest_path).parent

    with open(manifest_path) as f:
        manifest = json.load(f)

    chunk_names = manifest.get("chunk_files", [])
    if not chunk_names:
        print("ERROR: manifest has no chunk_files", file=sys.stderr)
        sys.exit(1)

    chunk_paths = [(manifest_dir / name).as_posix() for name in chunk_names]
    n_vols = len(chunk_paths)

    # Open FDs with O_RDONLY.
    fds = []
    for p in chunk_paths:
        fd = os.open(p, os.O_RDONLY)
        fds.append(fd)

    stats = [VolumeStats(path=chunk_paths[i]) for i in range(n_vols)]

    # Build a work schedule from the manifest entries — simulate fetching
    # one expert from each tensor entry, cycling through entries.
    work_items: list[tuple[int, int, int]] = []  # (fd_idx, offset, size)
    for entry in manifest.get("entries", []):
        for frag in entry.get("fragments", []):
            file_id = frag["file_id"]
            offset = frag["offset"]
            size = frag["size"]
            if size > 0 and file_id < n_vols:
                work_items.append((file_id, offset, size))

    if not work_items:
        # No fragments — fall back to raw reads at start of each file.
        file_sizes = [os.fstat(fd).st_size for fd in fds]
        for i in range(n_vols):
            block = min(DEFAULT_BLOCK_SIZE, file_sizes[i])
            if block > 0:
                work_items.append((i, 0, block))

    print(f"[nway-bench] Manifest: {manifest_path}")
    print(f"[nway-bench] Volumes: {n_vols}")
    print(f"[nway-bench] Work items per iteration: {len(work_items)}")
    print(f"[nway-bench] Iterations: {iterations} (+ {WARMUP_ITERATIONS} warmup)")
    print()

    # Warmup.
    for _ in range(WARMUP_ITERATIONS):
        for fd_idx, offset, size in work_items:
            os.pread(fds[fd_idx], size, offset)

    # Benchmark.
    if parallel:
        _bench_parallel(fds, work_items, iterations, stats)
    else:
        _bench_serial(fds, work_items, iterations, stats)

    for fd in fds:
        os.close(fd)

    return stats


def _bench_serial(fds, work_items, iterations, stats):
    """Serial benchmark: issues preads one at a time."""
    for it in range(iterations):
        for fd_idx, offset, size in work_items:
            result = do_pread(fds[fd_idx], size, offset, fd_idx)
            s = stats[fd_idx]
            s.ops += 1
            s.total_bytes += result.bytes_read
            s.total_us += result.latency_us
            s.latencies_us.append(result.latency_us)
            s.throughputs_gbps.append(result.throughput_gbps)


def _bench_parallel(fds, work_items, iterations, stats):
    """Parallel benchmark: fans out preads across volumes concurrently."""
    n_workers = max(4, len(fds) * 2)

    for it in range(iterations):
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for fd_idx, offset, size in work_items:
                futures.append(pool.submit(do_pread, fds[fd_idx], size, offset, fd_idx))

            for future in as_completed(futures):
                result = future.result()
                s = stats[result.fd_idx]
                s.ops += 1
                s.total_bytes += result.bytes_read
                s.total_us += result.latency_us
                s.latencies_us.append(result.latency_us)
                s.throughputs_gbps.append(result.throughput_gbps)


# ---------------------------------------------------------------------------
# Raw drive benchmark (no manifest)
# ---------------------------------------------------------------------------

def bench_raw_drives(
    drive_paths: list[str],
    block_size: int,
    iterations: int,
    file_size: int,
    parallel: bool,
) -> list[VolumeStats]:
    """Benchmark raw drive sequential read throughput."""
    n_vols = len(drive_paths)

    # Create temp benchmark files on each drive.
    bench_files: list[str] = []
    fds: list[int] = []
    try:
        for i, drive in enumerate(drive_paths):
            os.makedirs(drive, exist_ok=True)
            bench_file = os.path.join(drive, f"nway_bench_{i}.bin")
            bench_files.append(bench_file)

            # Write random data (page-aligned).
            print(f"[nway-bench] Creating bench file: {bench_file} ({file_size / (1024**2):.0f} MiB)")
            with open(bench_file, "wb") as f:
                remaining = file_size
                while remaining > 0:
                    chunk = min(remaining, 4 * 1024 * 1024)
                    f.write(os.urandom(chunk))
                    remaining -= chunk

            fd = os.open(bench_file, os.O_RDONLY)
            fds.append(fd)

        stats = [VolumeStats(path=drive_paths[i]) for i in range(n_vols)]

        # Build work items: sequential page-aligned reads through each file.
        n_blocks = file_size // block_size
        work_items: list[tuple[int, int, int]] = []
        for vol_idx in range(n_vols):
            for blk in range(n_blocks):
                offset = blk * block_size
                work_items.append((vol_idx, offset, block_size))

        print(f"\n[nway-bench] Drives: {n_vols}")
        print(f"[nway-bench] Block size: {block_size / 1024:.0f} KiB")
        print(f"[nway-bench] Blocks per file: {n_blocks}")
        print(f"[nway-bench] Total work items per iteration: {len(work_items)}")
        print(f"[nway-bench] Iterations: {iterations} (+ {WARMUP_ITERATIONS} warmup)")
        print()

        # Warmup.
        for _ in range(WARMUP_ITERATIONS):
            for fd_idx, offset, size in work_items[:min(len(work_items), n_vols * 4)]:
                os.pread(fds[fd_idx], size, offset)

        # Benchmark.
        if parallel:
            _bench_parallel(fds, work_items, iterations, stats)
        else:
            _bench_serial(fds, work_items, iterations, stats)

        for fd in fds:
            os.close(fd)

        return stats

    finally:
        # Clean up bench files.
        for bf in bench_files:
            try:
                os.unlink(bf)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_stats(stats: list[VolumeStats]) -> None:
    """Print per-volume statistics and recommended weights."""
    total_ops = sum(s.ops for s in stats)
    if total_ops == 0:
        print("[nway-bench] No operations recorded.")
        return

    print("=" * 100)
    print(f"{'Vol':>4}  {'Path':<40}  {'Ops':>8}  {'GiB':>8}  "
          f"{'Lat avg':>10}  {'Lat p50':>10}  {'Lat p99':>10}  "
          f"{'Thr agg':>10}  {'Thr p50':>10}  {'Thr p99':>10}")
    print(f"{'':>4}  {'':40}  {'':>8}  {'':>8}  "
          f"{'(us)':>10}  {'(us)':>10}  {'(us)':>10}  "
          f"{'(GB/s)':>10}  {'(GB/s)':>10}  {'(GB/s)':>10}")
    print("-" * 100)

    agg_throughputs = []

    for i, s in enumerate(stats):
        if s.ops == 0:
            print(f"{i:>4}  {s.path:<40}  {'(no ops)':>8}")
            agg_throughputs.append(0.0)
            continue

        avg_lat = s.total_us / s.ops
        p50_lat = percentile(s.latencies_us, 50.0)
        p99_lat = percentile(s.latencies_us, 99.0)
        total_gib = s.total_bytes / (1024 ** 3)
        agg_thr = s.total_bytes / (s.total_us * 1000.0) if s.total_us > 0 else 0.0
        p50_thr = percentile(s.throughputs_gbps, 50.0)
        p99_thr = percentile(s.throughputs_gbps, 99.0)

        agg_throughputs.append(agg_thr)

        print(f"{i:>4}  {s.path:<40}  {s.ops:>8}  {total_gib:>8.2f}  "
              f"{avg_lat:>10.1f}  {p50_lat:>10.1f}  {p99_lat:>10.1f}  "
              f"{agg_thr:>10.3f}  {p50_thr:>10.3f}  {p99_thr:>10.3f}")

    print("=" * 100)

    # Overall aggregate.
    total_bytes = sum(s.total_bytes for s in stats)
    total_us = max(s.total_us for s in stats) if stats else 0  # wall-clock is max of concurrent
    total_us_sum = sum(s.total_us for s in stats)
    total_gib = total_bytes / (1024 ** 3)

    # Serial aggregate (sum of times — no parallelism credit).
    serial_gbps = total_bytes / (total_us_sum * 1000.0) if total_us_sum > 0 else 0.0
    # Ideal parallel aggregate (limited by slowest volume).
    parallel_gbps = total_bytes / (total_us * 1000.0) if total_us > 0 else 0.0

    print(f"\n[nway-bench] Total: {total_gib:.2f} GiB across {total_ops} ops")
    print(f"[nway-bench] Aggregate throughput (serial sum): {serial_gbps:.3f} GB/s")
    print(f"[nway-bench] Aggregate throughput (parallel est): {parallel_gbps:.3f} GB/s")

    # Recommended weights.
    if any(t > 0 for t in agg_throughputs):
        # Normalize to the fastest drive.
        max_thr = max(agg_throughputs)
        if max_thr > 0:
            normalized = [t / max_thr for t in agg_throughputs]
        else:
            normalized = [1.0] * len(agg_throughputs)

        # Round to 1 decimal place for clean CLI usage.
        weights_str = " ".join(f"{w * max_thr:.2f}" for w in normalized)
        norm_str = " ".join(f"{w:.2f}" for w in normalized)

        print(f"\n[nway-bench] Recommended --weights (raw GB/s): {weights_str}")
        print(f"[nway-bench] Recommended --weights (normalized): {norm_str}")
        print(f"\n  Usage example:")
        print(f"    python tools/gguf-nway-split.py -i model.gguf -o ./nway-out --weights {weights_str}")

        # Check for imbalance.
        if len(normalized) > 1:
            slowest = min(normalized)
            if slowest < 0.5:
                print(f"\n  WARNING: Volume imbalance detected — slowest drive is "
                      f"{slowest:.0%} of the fastest.")
                print(f"  Consider removing the slowest volume or rebalancing the split.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-Way I/O Benchmark — evaluate multi-volume throughput and recommend weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark an existing nway split (reads chunk files):
  python tools/nway-bench.py --manifest ./nway-out/model-manifest.json

  # Benchmark raw drives (creates temp files, measures throughput):
  python tools/nway-bench.py --drives /Volumes/NVMe/bench /Volumes/TB4/bench

  # Increase iterations for more stable percentiles:
  python tools/nway-bench.py --manifest ./nway-out/model-manifest.json -n 500

  # Serial mode (no parallelism, measures pure single-drive speed):
  python tools/nway-bench.py --drives /Volumes/NVMe/bench --serial -n 1000
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--manifest", "-m",
        help="Path to model-manifest.json from gguf-nway-split.py",
    )
    mode.add_argument(
        "--drives", "-d",
        nargs="+",
        help="Paths to directories on each drive to benchmark (creates temp files)",
    )

    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of benchmark iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--block-size", "-b",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Read block size in bytes for --drives mode (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--file-size",
        type=int,
        default=DEFAULT_FILE_SIZE,
        help=f"Temp file size in bytes for --drives mode (default: {DEFAULT_FILE_SIZE})",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run reads serially instead of in parallel (measures single-drive speed)",
    )

    args = parser.parse_args()
    parallel = not args.serial

    if args.manifest:
        stats = bench_manifest(args.manifest, args.iterations, parallel)
    else:
        stats = bench_raw_drives(
            args.drives,
            args.block_size,
            args.iterations,
            args.file_size,
            parallel,
        )

    report_stats(stats)


if __name__ == "__main__":
    main()
