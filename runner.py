import csv
import math
import subprocess
import re
from pathlib import Path
from typing import Optional

import config


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, f"Command timed out: {' '.join(cmd)}"


def configure() -> tuple[bool, str]:
    return _run([
        config.CMAKE,
        "-S", str(config.CUDA_DIR),
        "-B", str(config.BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ])


def build(target: Optional[str] = None) -> tuple[bool, str]:
    return _run([
        config.CMAKE,
        "--build", str(config.BUILD_DIR),
        "--config", "Release",
    ] + (["--target", target] if target is not None else []))


def _validate_baseline_path(seed: int) -> Path:
    return config.REF_OUTPUT_FILE.with_name(f"{config.REF_OUTPUT_FILE.stem}_{seed}{config.REF_OUTPUT_FILE.suffix}")


def clear_validate_baselines() -> None:
    for s in range(config.NUM_SEEDS):
        _validate_baseline_path(s).unlink(missing_ok=True)


def run_validate(binary: Path, seed: int) -> tuple[bool, str]:
    return _run([str(binary), "--mode", "validate", "--seed", str(seed), str(_validate_baseline_path(seed))])


def run_validate_sweep(binary: Path) -> tuple[bool, Optional[float], str]:
    max_error: Optional[float] = None
    last_output = ""
    for s in range(config.NUM_SEEDS):
        ok, output = run_validate(binary, s)
        if not ok:
            return False, max_error, output
        last_output = output
        match = re.search(r"max_error=([\d.eE+-]+)", output)
        if match:
            err = float(match.group(1))
            max_error = err if max_error is None else max(max_error, err)
    return True, max_error, last_output


def run_benchmark(binary: Path, seed: Optional[int] = None) -> tuple[Optional[dict], str]:
    cmd = [
        str(binary),
        "--mode", "bench",
        "--warmup", str(config.BENCHMARK_WARMUP),
        "--timed", str(config.BENCHMARK_TIMED),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    ok, output = _run(cmd)
    if not ok:
        return None, output

    timing = {}
    for match in re.finditer(r"TIMING_(\w+):\s*(-?[\d.]+(?:e[-+]?\d+)?)", output):
        timing[match.group(1).lower()] = float(match.group(2))

    return timing if timing else None, output


def run_benchmark_sweep(binary: Path) -> tuple[Optional[dict], str]:
    """Benchmark over NUM_SEEDS configurations (each sampled from a seed)."""
    medians = []
    per_seed = []
    last_output = ""
    for s in range(config.NUM_SEEDS):
        timing, output = run_benchmark(binary, seed=s)
        if timing is None:
            return None, output
        last_output = output
        val = timing[config.BENCHMARK_METRIC]
        medians.append(val)
        per_seed.append((s, val))

    aggregate = math.prod(medians) ** (1.0 / len(medians))
    return {config.BENCHMARK_METRIC: aggregate, "per_seed": per_seed}, last_output


# NCU profiling

def ncu_capture(binary: Path, out_rep: Path) -> tuple[bool, str]:
    """Capture an NCU profile as .ncu-rep file."""
    cmd = [
        config.NCU,
        "--set", "full",
        "-f",
        "--import-source", "on",
        "-o", str(out_rep.with_suffix("")),
    ]
    cmd.extend([
        str(binary),
        "--mode", "bench",
        "--warmup", "0",
        "--timed", "1",
    ])
    return _run(cmd)


def ncu_query_metrics(rep_file: Path) -> str:
    """Query metrics from a .ncu-rep.

    Returns a structured summary grouped by kernel and section.
    """
    try:
        cmd = [
            config.NCU,
            "--import", str(rep_file),
            "--csv",
            "--page", "details",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        raw = result.stdout or ""
        lines = raw.strip().splitlines()
        if not lines:
            return raw

        reader = csv.DictReader(lines)
        groups: dict[tuple[int, str, str], list[dict[str, str]]] = {}
        action_order: list[tuple[int, str]] = []

        for row in reader:
            action_id = int(row.get("ID", "0"))
            kernel = row.get("Kernel Name", "")
            section = row.get("Section Name", "")
            key = (action_id, kernel, section)
            groups.setdefault(key, []).append(row)
            if (action_id, kernel) not in action_order:
                action_order.append((action_id, kernel))

        out: list[str] = []
        for aid, kernel in action_order:
            out.append(f"## {kernel} (ID {aid})")
            out.append("")
            for (gid, gkernel, gsection), rows in groups.items():
                if gid != aid:
                    continue
                # Skip sections where all rows are empty
                if not any(row.get("Metric Name", "") or row.get("Metric Value", "") for row in rows):
                    continue
                out.append(f"  ### {gsection}")
                for row in rows:
                    metric = row.get("Metric Name", "")
                    value = row.get("Metric Value", "")
                    if not metric and not value:
                        continue
                    unit = row.get("Metric Unit", "")
                    unit_str = f" {unit}" if unit else ""
                    out.append(f"  - {metric}: {value}{unit_str}")
                out.append("")

        result = "\n".join(out).rstrip()
        return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def ncu_query_source(rep_file: Path) -> str:
    """Line-annotated source for a metric from a .ncu-rep.

    Returns a cleaned TSV with SASS rows removed, constant columns dropped,
    and Line No removed.
    """
    try:
        cmd = [
            config.NCU,
            "--import", str(rep_file),
            "--csv",
            "--page", "source",
            "--print-source", "cuda,sass",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        raw = result.stdout or ""
        lines = raw.strip().splitlines()
        if not lines:
            return raw

        # First pass: remove SASS rows and metadata rows
        filtered_lines = []
        for line in lines:
            if line.startswith(('""', '"File Path"', '"Function Name"')):
                continue
            filtered_lines.append(line)

        reader = csv.reader(filtered_lines)
        all_rows = list(reader)
        if not all_rows:
            return raw

        # Split into kernel blocks: each block starts with a header row
        blocks: list[tuple[list[str], list[list[str]]]] = []
        current_header: list[str] | None = None
        current_rows: list[list[str]] = []

        for row in all_rows:
            # Check if this is a header row
            if "Line No" in row:
                if current_header is not None:
                    blocks.append((current_header, current_rows))
                current_header = row
                current_rows = []
            else:
                current_rows.append(row)

        if current_header is not None:
            blocks.append((current_header, current_rows))

        out: list[str] = []

        for header, rows in blocks:
            if not rows:
                continue

            col_count = len(header)
            # Find columns where all values are the same or "-"
            constant_cols: set[int] = set()
            # Always drop Line No
            if "Line No" in header:
                constant_cols.add(header.index("Line No"))

            for ci in range(col_count):
                if ci in constant_cols:
                    continue
                vals = set(row[ci].strip() if ci < len(row) else "" for row in rows)
                if len(vals) <= 1 or vals == {"-", ""}:
                    constant_cols.add(ci)

            # Build filtered header and rows
            keep_cols = [ci for ci in range(col_count) if ci not in constant_cols]
            filtered_header = [header[ci] for ci in keep_cols]
            filtered_rows = [
                [row[ci].strip() if ci < len(row) else "" for ci in keep_cols]
                for row in rows
            ]

            # Write as TSV without quotes
            out.append("\t".join(filtered_header))
            for fr in filtered_rows:
                out.append("\t".join(fr))
            out.append("")  # blank line between kernels

        result = "\n".join(out).rstrip()
        return result
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


# Git

def git(*args: str) -> tuple[bool, str]:
    """Thin wrapper around git subprocess."""
    return _run([config.GIT, *args])
