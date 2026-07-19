#!/usr/bin/env python3
# Description: Discovers every test_*.py in this directory, imports each module,
# and calls every test_* function it defines (plain-assert style, no pytest
# dependency). Prints a per-test PASS/FAIL/SKIP summary and exits nonzero on any
# failure. `--fast` skips functions named in each module's optional module-level
# `SLOW` set (multi-seed runs, anything that may construct a real torchvision
# backbone and could attempt a weight download).

import argparse
import importlib.util
import os
import sys
import traceback

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
THIS_FILE = os.path.basename(__file__)


def _discover_test_files():
    return sorted(
        f for f in os.listdir(TESTS_DIR)
        if f.startswith("test_") and f.endswith(".py") and f != THIS_FILE
    )


def _load_module(filename):
    name = filename[:-3]
    path = os.path.join(TESTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(fast=False):
    results = []  # (label, status in {"PASS", "FAIL", "SKIP"}, detail)

    for filename in _discover_test_files():
        try:
            module = _load_module(filename)
        except Exception as e:
            results.append((f"{filename} <import>", "FAIL", f"{type(e).__name__}: {e}"))
            traceback.print_exc()
            continue

        slow = getattr(module, "SLOW", set())
        test_names = sorted(
            name for name in dir(module)
            if name.startswith("test_") and callable(getattr(module, name))
        )

        for name in test_names:
            label = f"{filename}::{name}"
            if fast and name in slow:
                results.append((label, "SKIP", "--fast"))
                print(f"[SKIP] {label} (--fast)")
                continue
            try:
                getattr(module, name)()
                results.append((label, "PASS", ""))
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                results.append((label, "FAIL", detail))
                print(f"[FAIL] {label}: {detail}")
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, status, detail in results:
        suffix = f" -- {detail}" if status == "FAIL" else ""
        print(f"  [{status}] {label}{suffix}")

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped ({len(results)} total)")

    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                         help="Skip functions listed in each module's SLOW set")
    args = parser.parse_args()
    ok = run(fast=args.fast)
    sys.exit(0 if ok else 1)
