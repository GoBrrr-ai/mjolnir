The `sm_87` mma kernel — a CuTe-DSL port of NVIDIA's Ampere FlashAttention-2
example, restructured for paged decode — targets AGX Orin 64GB (16 SMs,
204.8 GB/s of LPDDR5, 4 MB L2) and serves fp8 and fp6 (`e2m3`) KV; it is
serving-validated and measured against the FlashInfer FA2-TC reference, with
the chapters quantifying where it stands. Open: the mma serves head_size 256
only (the hd128 gap is the top integration item), and PDL does not exist on
Ampere.
