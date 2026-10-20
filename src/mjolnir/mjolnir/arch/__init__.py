"""Implemented-kernel registry: one subpackage per supported architecture
(``mjolnir.arch.thor``).

``mjolnir.platforms.kernel_for`` maps a compute capability to a dotted module
path here. Importing a subpackage pulls the CUTLASS DSL (heavy) — only do it
when a device is real.
"""
