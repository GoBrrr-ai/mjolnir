"""fp6 expand canary (self-contained; the image's patched flashinfer + mjolnir).

Validates the `fi-fp6-kv-ampere-prefill.patch` expansion numerics end to end,
independently on BOTH sides:

  * the device side runs the patched flashinfer's ``expand_fp6_tile_to_fp8``
    (the in-smem e2m3 -> e4m3 rewrite the fp6 prefill fallback performs) on a
    tile packed per the kvfp6 row layout (4 codes per 3 bytes, LSB-first);
  * the host reference is built from ``mjolnir.kvfp6.e2m3_values`` (the shipped
    code table) cast through torch's float8_e4m3fn -- no shared code with the
    kernel's conversion path.

Bit-exact expected: e2m3 -> half is exact, the scale multiply is one fp32 op,
and both sides round to e4m3 saturating RNE. Checks the swizzled slot
addressing (k128B: ``i*stride + (j ^ (i % 8))``), the per-(kv_head, channel)
scale indexing, and the in-place write. Exit 0 iff every check passes;
self-skips without CUDA or off the boards the image targets.

The input tile layout (file/buffer order): CTA_TILE_KV rows x COLS b128
slots, slot (row, col) placed at the swizzled offset ``row*COLS + (col ^
(row % 8))``; each slot = 12 packed bytes + 4 zero bytes (the partial
cp.async's zero fill). Scales: HK x HD float32, row-major.
"""

import pathlib
import subprocess
import sys
import tempfile

import torch

H_K, D = 4, 256  # Qwen3.8-27B's KV heads / head dim
CTA_TILE_KV = 64
COLS = D // 16  # b128 slots per row: 16 codes (12 packed bytes) each
NSLOT = CTA_TILE_KV * COLS

CU = r"""
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "flashinfer/attention/prefill.cuh"

struct FakeTraits {
  using DTypeKV = flashinfer::mjolnir_fp6_e2m3_sv;
  static constexpr flashinfer::SwizzleMode SWIZZLE_MODE_KV = flashinfer::SwizzleMode::k128B;
  static constexpr uint32_t NUM_THREADS = 128;
  static constexpr uint32_t CTA_TILE_KV = 64;
};

constexpr uint32_t HD = 256, CTAK = 64, COLS = HD / 16, NSLOT = CTAK * COLS;

__global__ void expand(flashinfer::mjolnir_fp6_e2m3_sv* tile, const float* scales,
                       uint32_t head) {
  flashinfer::expand_fp6_tile_to_fp8<FakeTraits, HD>(tile, scales, head, threadIdx.x);
}

static std::vector<uint8_t> slurp(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(2); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> v(n);
  if (fread(v.data(), 1, n, f) != (size_t)n) { printf("short read %s\n", p); exit(2); }
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s in.bin scales.bin out.bin head\n", argv[0]); return 2; }
  auto in = slurp(argv[1]);
  auto sc = slurp(argv[2]);
  if (in.size() != NSLOT * 16 || sc.size() != 4 * HD * sizeof(float)) {
    printf("bad input sizes: in=%zu scales=%zu\n", in.size(), sc.size()); return 2;
  }
  uint32_t head = (uint32_t)atoi(argv[4]);

  flashinfer::mjolnir_fp6_e2m3_sv* d_tile;
  float* d_scales;
  if (cudaMalloc(&d_tile, NSLOT * 16) != cudaSuccess ||
      cudaMalloc(&d_scales, sc.size()) != cudaSuccess) { printf("cudaMalloc failed\n"); return 2; }
  cudaMemcpy(d_tile, in.data(), NSLOT * 16, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scales, sc.data(), sc.size(), cudaMemcpyHostToDevice);
  expand<<<1, FakeTraits::NUM_THREADS>>>(d_tile, d_scales, head);
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("expand failed: %s\n", cudaGetErrorString(err)); return 3; }

  std::vector<uint8_t> out(NSLOT * 16);
  cudaMemcpy(out.data(), d_tile, NSLOT * 16, cudaMemcpyDeviceToHost);
  FILE* f = fopen(argv[3], "wb");
  fwrite(out.data(), 1, out.size(), f);
  fclose(f);
  return 0;
}
"""


def _swizzled_slot_offset(row: int, col: int) -> int:
    # permuted_smem.cuh get_permuted_offset, k128B: i*stride + (j ^ (i % 8))
    return row * COLS + (col ^ (row % 8))


def _pack_slot(codes16: list[int]) -> bytes:
    # kvfp6 row layout: 4 codes per 3 bytes, LSB-first.
    out = bytearray(16)  # 12 packed + 4 zero (the partial cp.async fill)
    for g in range(4):
        w = 0
        for k in range(4):
            w |= (codes16[g * 4 + k] & 63) << (6 * k)
        out[g * 3 : g * 3 + 3] = w.to_bytes(3, "little")
    return bytes(out)


def _ref_byte(code: int, scale: float) -> int:
    v = float(E2M3_VALUES[code]) * scale
    return int(torch.tensor(v).to(torch.float8_e4m3fn).view(torch.uint8))


def main() -> int:
    if not torch.cuda.is_available():
        print("FP6 EXPAND: skipped (no CUDA)")
        return 0
    cc_major, _ = torch.cuda.get_device_capability()
    if cc_major < 8:
        print("FP6 EXPAND: skipped (pre-Ampere)")
        return 0

    from mjolnir.kvfp6 import e2m3_values

    global E2M3_VALUES
    E2M3_VALUES = e2m3_values(torch.device("cpu"))

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        # deterministic codes covering all 64 values across heads/rows/cols
        codes = torch.randint(0, 64, (H_K, CTA_TILE_KV, COLS * 16), generator=torch.Generator().manual_seed(7))
        scales = (torch.rand(H_K, D, generator=torch.Generator().manual_seed(3)) * 0.5 + 0.05).float()
        scales.numpy().tofile(td / "scales.bin")  # the full [H_K, D] table
        for head in range(H_K):
            in_tile = bytearray(NSLOT * 16)
            for row in range(CTA_TILE_KV):
                for col in range(COLS):
                    off = _swizzled_slot_offset(row, col) * 16
                    in_tile[off : off + 16] = _pack_slot(codes[head, row, col * 16 : (col + 1) * 16].tolist())
            (td / "in.bin").write_bytes(bytes(in_tile))

            cu = td / "fp6_expand_test.cu"
            cu.write_text(CU)
            exe = td / "fp6_expand_test"
            inc = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/data/include")
            cccl = inc.parent / "cccl"
            cmd = [
                "nvcc",
                "-std=c++17",
                str(cu),
                "-o",
                str(exe),
                f"-I{inc}",
                f"-I{cccl}/cub",
                f"-I{cccl}/libcudacxx/include",
                f"-I{cccl}/thrust",
                "--expt-relaxed-constexpr",
                "-arch=native",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FP6 EXPAND head {head}: COMPILE FAIL\n{r.stderr[-800:]}")
                return 1
            r = subprocess.run(
                [str(exe), str(td / "in.bin"), str(td / "scales.bin"), str(td / "out.bin"), str(head)],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(f"FP6 EXPAND head {head}: RUN FAIL {r.stdout}{r.stderr}")
                return 1

            out_tile = (td / "out.bin").read_bytes()
            bad = 0
            for row in range(CTA_TILE_KV):
                for col in range(COLS):
                    off = _swizzled_slot_offset(row, col) * 16
                    got = out_tile[off : off + 16]
                    want = bytes(
                        _ref_byte(int(codes[head, row, col * 16 + k]), float(scales[head, col * 16 + k]))
                        for k in range(16)
                    )
                    if got != want:
                        bad += 1
                        if bad <= 3:
                            print(f"  head {head} row {row} col {col}: got {got.hex()} want {want.hex()}")
            if bad:
                print(f"FP6 EXPAND head {head}: {bad} mismatched slots")
                return 1
            print(f"FP6 EXPAND head {head}: {CTA_TILE_KV * COLS} slots bit-exact")

    # kvfp6 cross-check: the code table the reference used must be the shipped one
    lut = e2m3_values(torch.device("cpu"))
    assert len(lut) == 64
    print("FP6 EXPAND: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
