Research platform (GB10, DGX OS + CUDA 13). No Mjolnir kernel targets
`sm_121` yet — the kernel registry ships `sm_110` and `sm_87` — so serving
here runs the stock backends and the platform's data is official-image
baselines only; the stock vLLM image stays the default until a kernel lands.
