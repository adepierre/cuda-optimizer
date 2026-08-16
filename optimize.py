import datetime
import json
import re
import shutil
import sys

import config
import llm
import runner


def plot_timing_history(history: list[dict], metric: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_val = history[0]["timing_ms"]
    best_val = min(
        h["timing_ms"][metric] if isinstance(h["timing_ms"], dict) else h["timing_ms"]
        for h in history if h.get("timing_ms") is not None
    )

    variants = {}

    timestamps = [h["timestamp"] for h in history]
    t0 = datetime.datetime.fromisoformat(timestamps[0])
    variants["time"] = {
        "out_path": config.LOGS_DIR / "timing_history.png",
        "xs": [0.0] + [
            (datetime.datetime.fromisoformat(t) - t0).total_seconds() / 3600 for t in timestamps[1:]
        ],
        "x_label": "Time (hours)",
        "title": "Optimization Timing History",
    }

    tokens_xs = [0]
    cumulative = 0
    for h in history[1:]:
        cumulative += (h.get("tokens") or {}).get("completion_tokens", 0)
        tokens_xs.append(cumulative / 1e6)
    variants["tokens"] = {
        "out_path": config.LOGS_DIR / "timing_history_tokens.png",
        "xs": tokens_xs,
        "x_label": "Cumulative output tokens (millions)",
        "title": "Optimization Timing History (by output tokens)",
    }

    points = {"accepted": [], "rejected": [], "rejected_capped": [], "failed": []}

    for i, h in enumerate(history):
        if h["status"] == "baseline":
            continue
        timing = h.get("timing_ms")
        if timing is None:
            points["failed"].append(i)
            continue
        val = timing[metric] if isinstance(timing, dict) else timing
        if h["status"] == "accepted":
            points["accepted"].append(i)
        elif val <= baseline_val:
            points["rejected"].append(i)
        else:
            points["rejected_capped"].append(i)

    for variant in variants.values():
        xs = variant["xs"]
        fig, ax = plt.subplots(figsize=(12, 5))

        def val(i):
            t = history[i]["timing_ms"]
            return t[metric] if isinstance(t, dict) else t

        all_vals = [baseline_val] + [val(i) for i in points["accepted"] + points["rejected"]]
        y_max = max(all_vals) * 1.15 if all_vals else baseline_val * 1.2
        y_min = min(all_vals) * 0.85

        ax.axhline(baseline_val, color="blue", linestyle="--", alpha=0.5, label=f"Baseline ({baseline_val:.2f} ms)")
        ax.axhline(best_val, color="purple", linestyle="--", alpha=0.5, label=f"Best ({best_val:.2f} ms)")

        accepted_xs = [xs[i] for i in points["accepted"]]
        ax.plot([xs[0]] + accepted_xs, [baseline_val] + [val(i) for i in points["accepted"]], "go-", linewidth=1.5, markersize=6, label="Accepted", zorder=3)

        rejected_xs = [xs[i] for i in points["rejected"]]
        if rejected_xs:
            ax.scatter(rejected_xs, [val(i) for i in points["rejected"]], marker="x", color="red", s=80, linewidths=1.5, label="Rejected", zorder=4)

        capped_xs = [xs[i] for i in points["rejected_capped"]]
        if capped_xs:
            ax.scatter(capped_xs, [y_max] * len(capped_xs), marker="x", color="red", s=80, linewidths=1.5, alpha=0.5, label="Worse than baseline", zorder=4)

        failed_xs = [xs[i] for i in points["failed"]]
        if failed_xs:
            ax.scatter(failed_xs, [baseline_val] * len(failed_xs), marker="o", facecolors="none", edgecolors="grey", s=50, linewidths=1.2, label="Failed", zorder=4)

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(variant["x_label"])
        ax.set_ylabel(f"Timing ({metric}) ms")
        ax.set_title(variant["title"])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(variant["out_path"], dpi=150)
        plt.close(fig)
        print(f"Timing graph saved to {variant['out_path']}")


def log_entry(entry: dict):
    """Append a JSON line to the log file."""
    entry["timestamp"] = datetime.datetime.now().isoformat()
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    # Startup
    # Create a new branch
    branch_name = f"{config.EXPERIMENT_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ok, _ = runner.git("checkout", "-b", branch_name)
    if not ok:
        print(f"Failed to create branch '{branch_name}'")
        sys.exit(1)
    print(f"Created branch: {branch_name}")

    print("Initializing optimized from reference...")
    shutil.copy2(
        config.REFERENCE_PATH,
        config.IMPL_PATH,
    )

    # Clean temp folders and stale snapshots
    if config.BUILD_DIR.exists():
        shutil.rmtree(config.BUILD_DIR)
    if config.LOGS_DIR.exists():
        shutil.rmtree(config.LOGS_DIR)

    print("Configuring CMake...")
    ok, output = runner.configure()
    if not ok:
        print(f"CMake configure failed:\n{output}")
        sys.exit(1)

    print("Building reference binary...")
    ok, output = runner.build(config.REFERENCE_TARGET)
    if not ok:
        print(f"Build failed:\n{output}")
        sys.exit(1)

    print("Writing baseline outputs...")
    runner.clear_validate_baselines()
    ok, _, output = runner.run_validate_sweep(config.REFERENCE_BIN)
    if not ok:
        match = re.search(r"max_error=([\d.eE+-]+)", output)
        max_err = float(match.group(1)) if match else None
        err_msg = f" (max_error={max_err})" if max_err is not None else ""
        print(f"Baseline validation failed{err_msg}:\n{output}")
        sys.exit(1)

    print("Building optimized binary...")
    ok, output = runner.build(config.OPTIMIZED_TARGET)
    if not ok:
        print(f"Build failed:\n{output}")
        sys.exit(1)

    print("Validating optimized binary...")
    ok, baseline_max_error, output = runner.run_validate_sweep(config.OPTIMIZED_BIN)
    if not ok:
        err_msg = f" (max_error={baseline_max_error})" if baseline_max_error is not None else ""
        print(f"Sanity validation failed{err_msg}:\n{output}")
        sys.exit(1)

    print("Running baseline benchmark...")
    baseline_timing, _ = runner.run_benchmark_sweep(config.OPTIMIZED_BIN)
    if baseline_timing is None:
        print("Baseline benchmark failed to parse timing.")
        sys.exit(1)

    best_timing = baseline_timing
    history = []

    print(f"Baseline timing: {best_timing}")
    baseline_entry = {"iter": 0, "status": "baseline", "timing_ms": best_timing[config.BENCHMARK_METRIC], "max_error": baseline_max_error}
    history.append(baseline_entry)
    log_entry(baseline_entry)

    ncu_rep = config.BUILD_DIR / "current.ncu-rep"
    print("Capturing NCU baseline profile...")
    ok, _ = runner.ncu_capture(config.OPTIMIZED_BIN, ncu_rep)

    has_profile = ok

    sys_prompt = llm.system_prompt()

    commits_made = 0
    no_improvement_count = 0

    for n in range(1, config.MAX_ITERATIONS + 1):
        # Re-profile only if code changed (accepted last iter)
        if not has_profile:
            print(f"[Iter {n}] Capturing NCU profile...")
            ok, _ = runner.ncu_capture(config.OPTIMIZED_BIN, ncu_rep)
            has_profile = ok

        retry_error = None
        last_max_error = None
        submit = None
        messages = None
        original_commit_message = None
        original_summary = None
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if has_profile:
            ncu_metrics = runner.ncu_query_metrics(ncu_rep)
        else:
            ncu_metrics = "No NCU profile available for this iteration."

        # Inner loop (retries)
        for retry in range(config.MAX_RETRIES):
            if messages is None:
                impl_source = config.IMPL_PATH.read_text()
                user_msg = llm.build_user_message(
                    impl_source, best_timing, history, ncu_metrics, retry_error
                )
                messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
            elif submit is not None:
                messages.append(submit.tool_call_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": submit.tool_call_id,
                    "content": llm.build_retry_error_message(retry_error),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Previous response timed out or failed. Error: {retry_error}. Please try to answer faster.",
                })

            print(f"[Iter {n}, retry {retry}] Calling LLM...")
            try:
                submit = llm.run_turn(messages)
                usage = submit.usage or {}
                failed = False
            except ValueError as e:
                retry_error = f"LLM response error: {e}"
                usage = {}
                failed = True
                print(f"[Iter {n}, retry {retry}] {retry_error}, retrying.")
            except:
                retry_error = f"Unknown error (LLM timeout?)"
                usage = {}
                failed = True
                print(f"[Iter {n}, retry {retry}] {retry_error}, retrying.")

            for k in tokens:
                tokens[k] += usage.get(k) or 0

            if failed:
                continue

            if original_commit_message is None:
                original_commit_message = submit.commit_message
                original_summary = submit.summary

            display_msg = submit.commit_message or original_commit_message or "(fix)"
            print(f"[Iter {n}, retry {retry}] LLM: {display_msg}")

            ok, error = llm.apply(submit)
            if not ok:
                retry_error = f"Apply failed: {error}"
                print(f"[Iter {n}, retry {retry}] Apply failed, retrying.")
                continue

            ok, output = runner.build()
            if not ok:
                retry_error = f"Build failed:\n{output[-2000:]}"
                print(f"[Iter {n}, retry {retry}] Build failed, retrying.")
                continue

            ok, last_max_error, output = runner.run_validate_sweep(config.OPTIMIZED_BIN)
            if not ok:
                err_detail = f" (max_error={last_max_error})" if last_max_error is not None else ""
                retry_error = f"Validation failed{err_detail}:\n{output[-2000:]}"
                print(f"[Iter {n}, retry {retry}] Validation failed{err_detail}, retrying.")
                continue

            break
        else:
            # All retries exhausted, revert and continue
            print(f"[Iter {n}] Retries exhausted, reverting.")
            runner.git("checkout", "--", str(config.IMPL_PATH))
            no_improvement_count += 1
            base_summary = original_summary or (submit.summary if submit else "no submission")
            failure_note = f' (failed: {retry_error.split("\n")[0]!r})' if retry_error else ""
            history.append({
                "iter": n,
                "status": "retries_exhausted",
                "timing_ms": None,
                "max_error": last_max_error,
                "summary": base_summary + failure_note,
                "tokens": tokens,
            })
            log_entry(history[-1])
            continue

        print(f"[Iter {n}] Benchmarking...")
        timing, _ = runner.run_benchmark_sweep(config.OPTIMIZED_BIN)
        if timing is None:
            print(f"[Iter {n}] Benchmark failed, reverting.")
            runner.git("checkout", "--", str(config.IMPL_PATH))
            no_improvement_count += 1
            history.append({
                "iter": n,
                "status": "benchmark_failed",
                "timing_ms": None,
                "max_error": last_max_error,
                "summary": original_summary or submit.summary,
                "tokens": tokens,
            })
            log_entry(history[-1])
            continue

        new_val = timing[config.BENCHMARK_METRIC]
        best_val = best_timing[config.BENCHMARK_METRIC]

        if new_val < best_val * (1 - config.IMPROVEMENT_THRESHOLD):
            runner.git("add", str(config.IMPL_PATH))
            runner.git("commit", "-m", original_commit_message or submit.commit_message)
            best_timing = timing
            has_profile = False  # Force re-profile next iter
            commits_made += 1
            no_improvement_count = 0
            history.append({
                "iter": n,
                "status": "accepted",
                "timing_ms": new_val,
                "max_error": last_max_error,
                "summary": original_summary or submit.summary,
                "tokens": tokens,
            })
            print(f"[Iter {n}] ACCEPTED: {new_val:.4f} ms (was {best_val:.4f} ms)")
        else:
            # Rejected
            runner.git("checkout", "--", str(config.IMPL_PATH))
            no_improvement_count += 1
            history.append({
                "iter": n,
                "status": "rejected",
                "timing_ms": new_val,
                "max_error": last_max_error,
                "summary": original_summary or submit.summary,
                "tokens": tokens,
            })
            print(f"[Iter {n}] REJECTED: {new_val:.4f} ms (no improvement over {best_val:.4f} ms)")

        log_entry(history[-1])

        if no_improvement_count >= config.MAX_NO_IMPROVEMENT:
            print(f"\nEarly stop: {no_improvement_count} consecutive iterations without improvement.")
            break

    plot_timing_history(history, config.BENCHMARK_METRIC)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total iterations:  {len(history) - 1}")
    print(f"Commits made:      {commits_made}")
    print(f"Metric used:       {config.BENCHMARK_METRIC}")
    print(f"Baseline timing:   {baseline_timing[config.BENCHMARK_METRIC]} ms")
    print(f"Best timing:       {best_timing[config.BENCHMARK_METRIC]} ms")

    if baseline_timing[config.BENCHMARK_METRIC] > 0:
        improvement = (
            1 - best_timing[config.BENCHMARK_METRIC] / baseline_timing[config.BENCHMARK_METRIC]
        ) * 100
        print(f"Improvement:       {improvement:.2f}%")


if __name__ == "__main__":
    main()
