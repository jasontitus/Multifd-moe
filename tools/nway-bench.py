#!/usr/bin/env python3
"""N-Way I/O Benchmark — mirrors the llama.cpp Flash-MoE slot-bank I/O path.

This tool replicates the *exact* I/O dispatch pattern used by the C++ runtime
in src/llama-context.cpp so that measured numbers predict real-world behavior:

  - Fixed 8-thread persistent pool (matches `start_read_pool()` n_workers=8)
  - Round-robin task distribution with stride=n_workers (matches the C++ worker
    loop: `for task_idx = idx; task_idx < num_tasks; task_idx += worker_count`)
  - Per-expert barrier: all fragments for one expert complete before the next
    expert is dispatched (matches `execute_pread_tasks()` + `work_done.wait()`)
  - pread() syscall via os.pread() — same kernel path as C `pread()`
  - FDs opened with O_RDONLY only — same flags as `llama-model.cpp` line 3040

Usage — benchmark existing nway split (reads the actual chunk files):
    python tools/nway-bench.py \\
        --manifest ./nway-out/model-manifest.json \\
        --iterations 200

Usage — benchmark raw drives (creates temp files, no manifest needed):
    python tools/nway-bench.py \\
        --drives /Volumes/NVMe/bench /Volumes/TB4_1/bench /Volumes/TB4_2/bench \\
        --block-size 2097152 \\
        --iterations 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — match the C++ runtime exactly
# ---------------------------------------------------------------------------
PAGE_SIZE = 16384             # 16 KB — Apple Silicon page alignment
DEFAULT_THREADS_PER_DEVICE = 2  # matches LLAMA_FLASH_MOE_READ_POOL_THREADS_PER_DEVICE default
BASE_POOL_WORKERS = 8         # floor for non-nway case (matches C++ `constexpr size_t base = 8`)
DEFAULT_BLOCK_SIZE = 2 * 1024 * 1024   # 2 MiB — typical expert tensor slice
DEFAULT_ITERATIONS = 200
DEFAULT_FILE_SIZE = 256 * 1024 * 1024  # 256 MiB bench file per drive
WARMUP_ITERATIONS = 10


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PreadTask:
    """Mirrors the C++ pread_task struct."""
    fd: int
    fd_idx: int        # volume index for stats attribution
    offset: int
    size: int
    result: int = 0
    elapsed_us: float = 0.0


@dataclass
class VolumeStats:
    """Per-volume accumulated statistics."""
    path: str
    ops: int = 0
    total_bytes: int = 0
    total_us: float = 0.0
    latencies_us: list[float] = field(default_factory=list)
    throughputs_gbps: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Thread pool that mirrors the C++ read_thread_pool
#
# The C++ pool uses:
#   - A fixed set of persistent worker threads
#   - A shared tasks pointer + count set by the master
#   - A generation counter bumped per batch
#   - Each worker handles tasks[idx], tasks[idx + n_workers], tasks[idx + 2*n_workers], ...
#   - Workers signal done via tasks_completed counter + condition variable
#   - Master waits on work_done until all workers report in
# ---------------------------------------------------------------------------

def compute_pool_workers(n_volumes: int, threads_per_device: int = DEFAULT_THREADS_PER_DEVICE) -> int:
    """Compute read pool worker count — mirrors effective_read_pool_workers() in C++."""
    env_val = os.environ.get("LLAMA_FLASH_MOE_READ_POOL_WORKERS", "")
    if env_val.strip().isdigit() and int(env_val) > 0:
        return int(env_val)
    env_tpd = os.environ.get("LLAMA_FLASH_MOE_READ_POOL_THREADS_PER_DEVICE", "")
    if env_tpd.strip().isdigit() and int(env_tpd) > 0:
        threads_per_device = int(env_tpd)
    return max(BASE_POOL_WORKERS, n_volumes * threads_per_device)


class ReadThreadPool:
    """Python replica of the C++ read_thread_pool in llama-context.cpp."""

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self._tasks: list[PreadTask] = []
        self._generation = 0
        self._completed_generation = 0
        self._tasks_completed = 0
        self._shutdown = False

        self._mutex = threading.Lock()
        self._work_ready = threading.Condition(self._mutex)
        self._work_done = threading.Condition(self._mutex)

        self._workers: list[threading.Thread] = []
        for idx in range(n_workers):
            t = threading.Thread(target=self._worker_fn, args=(idx,), daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_fn(self, idx: int) -> None:
        """Worker loop — mirrors the C++ lambda in start_read_pool()."""
        my_generation = 0
        while True:
            # Wait for work.
            with self._work_ready:
                self._work_ready.wait_for(
                    lambda: self._shutdown or self._generation != my_generation
                )
                if self._shutdown:
                    return
                my_generation = self._generation
                tasks = self._tasks
                num_tasks = len(tasks)

            # Execute tasks with round-robin stride — exactly matching C++:
            #   for (int task_idx = int(idx); task_idx < num_tasks; task_idx += worker_count)
            worker_count = self.n_workers
            task_idx = idx
            while task_idx < num_tasks:
                task = tasks[task_idx]
                t0 = time.monotonic()
                data = os.pread(task.fd, task.size, task.offset)
                task.elapsed_us = (time.monotonic() - t0) * 1e6
                task.result = len(data)
                task_idx += worker_count

            # Signal completion.
            with self._mutex:
                self._tasks_completed += 1
                if self._tasks_completed == self.n_workers:
                    self._completed_generation = my_generation
                    self._work_done.notify_all()

    def execute(self, tasks: list[PreadTask]) -> None:
        """Dispatch a batch of tasks and wait for completion (per-expert barrier)."""
        if not tasks:
            return

        with self._mutex:
            self._tasks = tasks
            self._tasks_completed = 0
            self._generation += 1
            generation = self._generation
            self._work_ready.notify_all()

        with self._work_done:
            self._work_done.wait_for(
                lambda: self._completed_generation >= generation or self._shutdown
            )

    def shutdown(self) -> None:
        with self._mutex:
            self._shutdown = True
            self._work_ready.notify_all()
        for w in self._workers:
            w.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(pct / 100.0 * len(s)), len(s) - 1)
    return s[idx]


def record_task_stats(tasks: list[PreadTask], stats: list[VolumeStats]) -> None:
    """Attribute completed task metrics to per-volume stats."""
    for task in tasks:
        s = stats[task.fd_idx]
        s.ops += 1
        actual_bytes = task.result if task.result > 0 else 0
        s.total_bytes += actual_bytes
        s.total_us += task.elapsed_us
        s.latencies_us.append(task.elapsed_us)
        if task.elapsed_us > 0 and actual_bytes > 0:
            gbps = actual_bytes / (task.elapsed_us * 1000.0)
            s.throughputs_gbps.append(gbps)


# ---------------------------------------------------------------------------
# Expert-level work schedule builder
#
# In the real runtime, the slot-bank fetches one expert at a time:
#   1. Look up the expert's fragments across chunk files
#   2. Build pread_task array (one per fragment)
#   3. execute_pread_tasks() — barrier until all fragments complete
#   4. Next expert
#
# We replicate this by grouping work_items into "expert batches".
# ---------------------------------------------------------------------------

@dataclass
class ExpertBatch:
    """One expert fetch = a batch of fragments dispatched together with a barrier."""
    tasks: list[PreadTask] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manifest-based benchmark
# ---------------------------------------------------------------------------

def bench_manifest(
    manifest_path: str,
    iterations: int,
    parallel: bool,
) -> list[VolumeStats]:
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

    fds = []
    for p in chunk_paths:
        fd = os.open(p, os.O_RDONLY)
        fds.append(fd)

    stats = [VolumeStats(path=chunk_paths[i]) for i in range(n_vols)]

    # Build expert batches from the manifest.
    # Each manifest entry = one tensor.  Each tensor has N fragments.
    # In the runtime, a single expert fetch reads a slice of each fragment.
    # For benchmarking, we treat each entry's full fragment set as one expert batch.
    expert_batches: list[ExpertBatch] = []
    for entry in manifest.get("entries", []):
        batch = ExpertBatch()
        for frag in entry.get("fragments", []):
            file_id = frag["file_id"]
            offset = frag["offset"]
            size = frag["size"]
            if size > 0 and file_id < n_vols:
                batch.tasks.append(PreadTask(
                    fd=fds[file_id], fd_idx=file_id,
                    offset=offset, size=size,
                ))
        if batch.tasks:
            expert_batches.append(batch)

    if not expert_batches:
        file_sizes = [os.fstat(fd).st_size for fd in fds]
        batch = ExpertBatch()
        for i in range(n_vols):
            block = min(DEFAULT_BLOCK_SIZE, file_sizes[i])
            if block > 0:
                batch.tasks.append(PreadTask(
                    fd=fds[i], fd_idx=i, offset=0, size=block,
                ))
        if batch.tasks:
            expert_batches.append(batch)

    n_pool = compute_pool_workers(n_vols)
    total_tasks = sum(len(b.tasks) for b in expert_batches)
    print(f"[nway-bench] Manifest:        {manifest_path}")
    print(f"[nway-bench] Volumes:         {n_vols}")
    print(f"[nway-bench] Expert batches:  {len(expert_batches)}")
    print(f"[nway-bench] Total tasks:     {total_tasks}")
    print(f"[nway-bench] Pool workers:    {n_pool} ({DEFAULT_THREADS_PER_DEVICE}/device, matching C++ runtime)")
    print(f"[nway-bench] Dispatch:        round-robin stride={n_pool}, per-expert barrier")
    print(f"[nway-bench] Iterations:      {iterations} (+ {WARMUP_ITERATIONS} warmup)")
    print()

    _run_benchmark(fds, expert_batches, iterations, stats, parallel, n_pool)

    for fd in fds:
        os.close(fd)

    return stats


# ---------------------------------------------------------------------------
# Raw drive benchmark
# ---------------------------------------------------------------------------

def bench_raw_drives(
    drive_paths: list[str],
    block_size: int,
    iterations: int,
    file_size: int,
    parallel: bool,
) -> list[VolumeStats]:
    """Benchmark raw drive throughput with synthetic expert batches."""
    n_vols = len(drive_paths)

    bench_files: list[str] = []
    fds: list[int] = []
    try:
        for i, drive in enumerate(drive_paths):
            os.makedirs(drive, exist_ok=True)
            bench_file = os.path.join(drive, f"nway_bench_{i}.bin")
            bench_files.append(bench_file)

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

        # Build expert batches: each "expert" reads one block from each volume
        # (simulates a tensor split across all drives).
        n_blocks = file_size // block_size
        expert_batches: list[ExpertBatch] = []
        for blk in range(n_blocks):
            batch = ExpertBatch()
            offset = blk * block_size
            for vol_idx in range(n_vols):
                batch.tasks.append(PreadTask(
                    fd=fds[vol_idx], fd_idx=vol_idx,
                    offset=offset, size=block_size,
                ))
            expert_batches.append(batch)

        n_pool = compute_pool_workers(n_vols)
        total_tasks = sum(len(b.tasks) for b in expert_batches)
        print(f"\n[nway-bench] Drives:          {n_vols}")
        print(f"[nway-bench] Block size:      {block_size / 1024:.0f} KiB")
        print(f"[nway-bench] Expert batches:  {len(expert_batches)} (1 per block offset)")
        print(f"[nway-bench] Total tasks:     {total_tasks}")
        print(f"[nway-bench] Pool workers:    {n_pool} ({DEFAULT_THREADS_PER_DEVICE}/device, matching C++ runtime)")
        print(f"[nway-bench] Dispatch:        round-robin stride={n_pool}, per-expert barrier")
        print(f"[nway-bench] Iterations:      {iterations} (+ {WARMUP_ITERATIONS} warmup)")
        print()

        _run_benchmark(fds, expert_batches, iterations, stats, parallel, n_pool)

        for fd in fds:
            os.close(fd)

        return stats

    finally:
        for bf in bench_files:
            try:
                os.unlink(bf)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def _run_benchmark(
    fds: list[int],
    expert_batches: list[ExpertBatch],
    iterations: int,
    stats: list[VolumeStats],
    parallel: bool,
    n_pool_workers: int,
) -> None:
    """Run the benchmark with the same dispatch semantics as the C++ runtime."""

    if parallel:
        pool = ReadThreadPool(n_workers=n_pool_workers)
    else:
        pool = None

    try:
        # Warmup — run through all expert batches serially to prime OS page cache.
        for _ in range(WARMUP_ITERATIONS):
            for batch in expert_batches:
                for task in batch.tasks:
                    os.pread(task.fd, task.size, task.offset)

        # Main benchmark loop.
        for it in range(iterations):
            for batch in expert_batches:
                # Reset task results for this dispatch.
                for task in batch.tasks:
                    task.result = 0
                    task.elapsed_us = 0.0

                if pool is not None:
                    # Parallel: dispatch through the persistent 8-thread pool
                    # with round-robin distribution and per-expert barrier.
                    pool.execute(batch.tasks)
                else:
                    # Serial: execute one task at a time (same as C++ fallback
                    # when read_pool.workers is empty).
                    for task in batch.tasks:
                        t0 = time.monotonic()
                        data = os.pread(task.fd, task.size, task.offset)
                        task.elapsed_us = (time.monotonic() - t0) * 1e6
                        task.result = len(data)

                # Record per-FD stats (mirrors record_nway_pread_stats).
                record_task_stats(batch.tasks, stats)

    finally:
        if pool is not None:
            pool.shutdown()


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

    total_bytes = sum(s.total_bytes for s in stats)
    total_us_max = max(s.total_us for s in stats) if stats else 0
    total_us_sum = sum(s.total_us for s in stats)
    total_gib = total_bytes / (1024 ** 3)

    serial_gbps = total_bytes / (total_us_sum * 1000.0) if total_us_sum > 0 else 0.0
    parallel_gbps = total_bytes / (total_us_max * 1000.0) if total_us_max > 0 else 0.0

    print(f"\n[nway-bench] Total: {total_gib:.2f} GiB across {total_ops} ops")
    print(f"[nway-bench] Aggregate throughput (serial sum): {serial_gbps:.3f} GB/s")
    print(f"[nway-bench] Aggregate throughput (parallel est): {parallel_gbps:.3f} GB/s")

    # Recommended weights.
    if any(t > 0 for t in agg_throughputs):
        max_thr = max(agg_throughputs)
        weights_str = " ".join(f"{t:.2f}" for t in agg_throughputs)
        norm_str = " ".join(f"{t / max_thr:.2f}" if max_thr > 0 else "1.00"
                            for t in agg_throughputs)

        print(f"\n[nway-bench] Recommended --weights (raw GB/s): {weights_str}")
        print(f"[nway-bench] Recommended --weights (normalized): {norm_str}")
        print(f"\n  Usage example:")
        print(f"    python tools/gguf-nway-split.py -i model.gguf -o ./nway-out --weights {weights_str}")

        if len(agg_throughputs) > 1:
            slowest = min(t for t in agg_throughputs if t > 0) if any(t > 0 for t in agg_throughputs) else 0
            if max_thr > 0 and slowest / max_thr < 0.5:
                print(f"\n  WARNING: Volume imbalance detected — slowest drive is "
                      f"{slowest / max_thr:.0%} of the fastest.")
                print(f"  Consider removing the slowest volume or rebalancing the split.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-Way I/O Benchmark — mirrors the llama.cpp Flash-MoE runtime I/O path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
I/O dispatch fidelity (matches src/llama-context.cpp):
  - 8 persistent worker threads (start_read_pool, n_workers=8)
  - Round-robin task assignment with stride=8
  - Per-expert completion barrier (execute_pread_tasks)
  - pread() syscall, FDs opened with O_RDONLY only

Examples:
  # Benchmark an existing nway split:
  python tools/nway-bench.py --manifest ./nway-out/model-manifest.json

  # Benchmark raw drives before repacking:
  python tools/nway-bench.py --drives /Volumes/NVMe/bench /Volumes/TB4/bench

  # More iterations for stable percentiles:
  python tools/nway-bench.py --manifest ./nway-out/model-manifest.json -n 500

  # Serial mode (no thread pool, measures pure single-drive speed):
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
        help=f"Temp file size for --drives mode in bytes (default: {DEFAULT_FILE_SIZE})",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run reads serially (no thread pool) — matches C++ fallback when pool is empty",
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
