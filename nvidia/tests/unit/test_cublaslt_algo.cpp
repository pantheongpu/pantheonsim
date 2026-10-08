// cuBLASLt's algorithm configuration, which PyTorch 2.14 binds at load time
// and a tuner uses: init, set and read back attributes, check against layouts.
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "vtest.hpp"

VTEST(init_set_get_round_trip) {
  cublasLtMatmulAlgo_t algo;
  VCHECK_EQ(cublasLtMatmulAlgoInit(nullptr, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                   CUDA_R_32F, 6, &algo), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(cublasLtMatmulAlgoInit(nullptr, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                   CUDA_R_32F, 6, nullptr), CUBLAS_STATUS_INVALID_VALUE);
  uint32_t v = 0;
  size_t n = 0;
  VCHECK_EQ(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &v, sizeof v, &n), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(v, 6u);
  VCHECK_EQ(n, sizeof v);
  const uint32_t tile = 20, splitk = 3;
  VCHECK_EQ(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk), CUBLAS_STATUS_SUCCESS);
  const uint16_t shape = 5;
  VCHECK_EQ(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &shape, sizeof shape), CUBLAS_STATUS_SUCCESS);
  v = 0;
  VCHECK_EQ(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &v, sizeof v, &n), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(v, 20u);
  VCHECK_EQ(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &v, sizeof v, &n), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(v, 3u);
  uint16_t s2 = 0;
  VCHECK_EQ(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &s2, sizeof s2, &n), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(s2, 5);
  VCHECK_EQ(n, sizeof s2);
  // the id was not disturbed by the others
  VCHECK_EQ(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &v, sizeof v, &n), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(v, 6u);
  // a wrong size, an unknown attribute, a size query
  VCHECK_EQ(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 2), CUBLAS_STATUS_INVALID_VALUE);
  VCHECK_EQ(cublasLtMatmulAlgoConfigSetAttribute(&algo, static_cast<cublasLtMatmulAlgoConfigAttributes_t>(99), &tile, 4),
            CUBLAS_STATUS_INVALID_VALUE);
  VCHECK_EQ(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, nullptr, 0, &n), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(n, sizeof tile);
}

VTEST(check_accepts_what_the_matmul_accepts_and_keeps_the_algo) {
  cublasLtHandle_t h = nullptr;
  VCHECK_EQ(cublasLtCreate(&h), CUBLAS_STATUS_SUCCESS);
  cublasLtMatmulDesc_t desc = nullptr;
  VCHECK_EQ(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  VCHECK_EQ(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, 8, 4, 8), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, 4, 6, 4), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, 8, 6, 8), CUBLAS_STATUS_SUCCESS);
  cublasLtMatmulAlgo_t algo;
  VCHECK_EQ(cublasLtMatmulAlgoInit(h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 1,
                                   &algo), CUBLAS_STATUS_SUCCESS);
  cublasLtMatmulHeuristicResult_t r;
  r.algo = algo;
  r.workspaceSize = 123;
  VCHECK_EQ(cublasLtMatmulAlgoCheck(h, desc, a, b, c, c, &algo, &r), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(r.state, CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(r.workspaceSize, 0u);
  VCHECK(r.wavesCount == 1.0f);
  // the shapes do not multiply: 8x4 times 8x6
  cublasLtMatrixLayout_t bad = nullptr;
  VCHECK_EQ(cublasLtMatrixLayoutCreate(&bad, CUDA_R_32F, 8, 6, 8), CUBLAS_STATUS_SUCCESS);
  VCHECK(cublasLtMatmulAlgoCheck(h, desc, a, bad, c, c, &algo, &r) != CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(cublasLtMatmulAlgoCheck(h, desc, a, b, c, c, nullptr, &r), CUBLAS_STATUS_INVALID_VALUE);
  cublasLtMatrixLayoutDestroy(bad);
  cublasLtMatrixLayoutDestroy(a);
  cublasLtMatrixLayoutDestroy(b);
  cublasLtMatrixLayoutDestroy(c);
  cublasLtMatmulDescDestroy(desc);
  cublasLtDestroy(h);
}

VTEST_MAIN
