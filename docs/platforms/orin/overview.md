The `sm_87` mma kernel — a CuTe-DSL port of NVIDIA's Ampere FlashAttention-2
example, restructured for paged decode — targets AGX Orin 64GB (16 SMs,
204.8 GB/s of LPDDR5, 4 MB L2) and serves fp8 and fp6 (`e2m3`) KV; it is
serving-validated and measured against the FlashInfer FA2-TC reference, with
the chapters quantifying where it stands. hd128 shipped — the mma serves
fp8 KV at head dim 128 and 256 from one D-parameterized body (kernel,
runner, and the split-KV combine all take the head size). Open: fp6 stays
hd256-only (the e2m3 recipe's static byte-lane phase assumes 64 groups per
row), and PDL does not exist on Ampere.
