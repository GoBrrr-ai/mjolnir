"""The kernel tests need a CUDA box with torch + the CUTLASS DSL (they run in
the Mjolnir image via `mjolnir verify vllm`); plain CI collects nothing here."""

import importlib.util

collect_ignore = ["smoke_vllm.py"]  # a script, not a pytest module
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("cutlass") is None:
    collect_ignore.append("test_runner.py")
