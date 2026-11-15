// The motivating example of the FTTN paper (Li et al., CCGrid 2024, arXiv 2403.00232, section V): D = C - A x B with A and B in
// binary16 and C and D in binary32, K = 2^13, every C = 2^20, A = 2^10 in column 0 and 2^-2 / 2^-3 in the odd / even columns, B = 2^10 in
// row 0 and -2^-3 elsewhere. The paper reports 0 on V100, A100 and MI250X, 255.875 on MI100 and 191.875 on H100 (the real-number answer
// is 2^7 + 2^6 - 2^-6 = 191.984375).
//
// FTTN's four programs never call the GEMM it ships for this (init_host_matrices and simple_wmma_run in its utility headers), so this file
// is the driver: it includes FTTN's own header and calls those two functions unchanged. Built with fttn.patch applied, which only lets
// M_TILES and N_TILES be set from the command line: every element of D is the same sum, so one 16 x 16 tile of outputs (M = N = 16) gives
// the value the paper's 8192 x 8192 outputs give, and K stays 2^13. Set -DM_TILES=512 -DN_TILES=512 for the paper's full size on a card.
//
// This file is not part of FTTN; it is under this repository's licence.
#include "utility/common.hpp"

#include <cstdio>
#include <cstring>

int main() {
  DeviceProp prop;
  getDeviceProperties(&prop, 0);
  const size_t na = size_t{M_GLOBAL} * K_GLOBAL, nb = size_t{N_GLOBAL} * K_GLOBAL, nc = size_t{M_GLOBAL} * N_GLOBAL;
  FP16 *h_a = new FP16[na], *h_b = new FP16[nb], *d_a, *d_b;
  FP32 *h_c = new FP32[nc], *h_d = new FP32[nc], *d_c, *d_d;
  CHECK_DEVICE_ERROR(deviceMalloc((void**)&d_a, na * sizeof(FP16)));
  CHECK_DEVICE_ERROR(deviceMalloc((void**)&d_b, nb * sizeof(FP16)));
  CHECK_DEVICE_ERROR(deviceMalloc((void**)&d_c, nc * sizeof(FP32)));
  CHECK_DEVICE_ERROR(deviceMalloc((void**)&d_d, nc * sizeof(FP32)));
  init_host_matrices(h_a, h_b, h_c, h_d);
  simple_wmma_run(h_a, h_b, h_c, h_d, d_a, d_b, d_c, d_d, -1.0f, 1.0f);
  CHECK_LAST_DEVICE_ERROR();
  size_t differ = 0;
  for (size_t t = 1; t < nc; ++t) differ += std::memcmp(&h_d[t], &h_d[0], sizeof(FP32)) != 0;
  uint32_t bits;
  std::memcpy(&bits, &h_d[0], sizeof bits);
  std::printf("MOTIVATING gpu=\"%s\" m=%d n=%d k=%d D=%.6f bits=0x%08x elements=%zu differ_from_first=%zu\n", prop.name, M_GLOBAL, N_GLOBAL, K_GLOBAL,
              static_cast<double>(h_d[0]), bits, nc, differ);
  return differ != 0;
}
