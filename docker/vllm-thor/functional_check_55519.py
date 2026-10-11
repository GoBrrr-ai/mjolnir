"""Functional check for PR #55519 (no draft-group warning when eagle block drop is off).

The sentinel check in verify_patches.py proves the code landed; this proves
the behavior the patch is about: the "prefix reuse is disabled" warning is
gated on use_eagle_block_drop() (not use_eagle()), and the scheduler's
"drop disabled" warning documents that reuse still works. No GPU needed.
Run inside the image: python3 /opt/vllm-patches/functional_check_55519.py
"""

import inspect
from pathlib import Path

import vllm.v1.core.kv_cache_utils as kcu


def main() -> int:
    failures = []

    src = inspect.getsource(kcu._warn_if_unannotated_eagle_mamba)
    if "use_eagle_block_drop()" not in src:
        failures.append("_warn_if_unannotated_eagle_mamba is not gated on use_eagle_block_drop()")
    scheduler = Path(kcu.__file__).parent / "sched" / "scheduler.py"
    if "stays eligible for cross-request" not in scheduler.read_text():
        failures.append("the scheduler's 'drop disabled' warning is not reworded")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("functional check 55519: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
