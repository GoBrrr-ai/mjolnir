"""The kernel tests need a CUDA box with torch + the CUTLASS DSL (they run in
the Mjolnir image via `mjolnir verify vllm`); plain CI collects nothing here.
Each family's runner suite is collected on its own architecture only:
test_runner.py on Thor (sm_110), test_orin_runner.py on Orin (sm_87)."""

import importlib.util

collect_ignore = ["smoke_vllm.py"]  # a script, not a pytest module


def _capability():
    import torch

    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


if importlib.util.find_spec("torch") is None or importlib.util.find_spec("cutlass") is None:
    collect_ignore += ["test_runner.py", "test_orin_runner.py"]
else:
    _cc = _capability()
    if _cc != (8, 7):
        collect_ignore.append("test_orin_runner.py")
    if _cc == (8, 7):
        collect_ignore.append("test_runner.py")  # the Thor (tcgen05) suite
