"""The Orin (``sm_87``) kernel family: the GEMV FMA decode kernel.

One subpackage per implemented architecture, sibling to
:mod:`mjolnir.arch.thor` (see that module's docstring for the surface
contract). Unlike Thor — whose kernel is a single traced file re-exported
through a module — the Orin family is a subpackage:

* :mod:`mjolnir.arch.orin.gemv` — the restored + retargeted GEMV FMA
  kernel (the seed of the family; targets ``sm_87`` only);
* the runner, split-KV plan and fp6 reader land here next — the surface
  the vLLM backend consumes (``Runner`` + the shared
  :mod:`mjolnir.runner` helpers) is not exported yet, and
  ``mjolnir.platforms._SHIPPED`` still marks orin as unimplemented until
  it exists.

Do not import this package on a device whose compute capability is not
(8, 7): the DSL compile and the module's budget asserts assume GA10B.
"""
