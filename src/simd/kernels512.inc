// The tile-product kernels (simd::Kernels::tile_f32, tile_i32, tile_f64) at the AVX-512 level: simd.cpp includes this file inside a namespace of
// its own, under #pragma GCC target for AVX-512 F/BW/DQ/VL/VNNI, and the Avx512 table is the Avx2 one with these three entries replaced.
//
// Every kernel gives the bits of the AVX2 and baseline ones (tests/unit/test_tile_mma.cpp runs each against a scalar loop at every level the CPU
// has): the same order of summation (k in order, each element its own chain, the products of the f32 kernel rounded and then added -- never fused
// -- and the f64 steps fused), and, for the integers, the same wrapping sum, which a 16-bit dot-product instruction (VNNI) reaches by another route
// (a sum of exact products, modulo 2^32, does not depend on the order) when every operand is a 16-bit number and the plain multiplier otherwise.
//
// Contraction is off for the whole file (simd.cpp's pragma): a product the card rounds before the sum must never become an FMA.

typedef float V16f __attribute__((vector_size(64)));
typedef uint32_t V16u __attribute__((vector_size(64)));
typedef int32_t V16i __attribute__((vector_size(64)));
typedef float V8f_ __attribute__((vector_size(32)));
typedef uint32_t V8u_ __attribute__((vector_size(32)));
typedef int32_t V8i_ __attribute__((vector_size(32)));

// R rows by V vectors of L columns (VT, L floats) of D = C + A x B in f32; as the AVX2 kernel's block (kernels.inc), at another width.
template <class VT, unsigned L, unsigned R, unsigned V>
inline void tile_f32_blockw(unsigned i0, unsigned j0, unsigned N, unsigned K, const float* A, const float* B, const float* C, size_t ldc, float* D, size_t ldd) {
  VT acc[R][V];
  for (unsigned r = 0; r < R; ++r)
    for (unsigned v = 0; v < V; ++v) {
      if (C) std::memcpy(&acc[r][v], C + size_t{i0 + r} * ldc + j0 + L * v, sizeof(VT));
      else acc[r][v] = VT{};
    }
  for (unsigned k = 0; k < K; ++k) {
    VT b[V];
    for (unsigned v = 0; v < V; ++v) std::memcpy(&b[v], B + size_t{k} * N + j0 + L * v, sizeof(VT));
    for (unsigned r = 0; r < R; ++r) {
      const float a = A[size_t{i0 + r} * K + k];
      for (unsigned v = 0; v < V; ++v) {
        const VT p = a * b[v];
        acc[r][v] = acc[r][v] + p;
      }
    }
  }
  for (unsigned r = 0; r < R; ++r)
    for (unsigned v = 0; v < V; ++v) std::memcpy(D + size_t{i0 + r} * ldd + j0 + L * v, &acc[r][v], sizeof(VT));
}

void tile_f32_k(unsigned M, unsigned N, unsigned K, const float* A, const float* B, const float* C, size_t ldc, float* D, size_t ldd) {
  unsigned i0 = 0;
  for (; i0 + 4 <= M; i0 += 4) {
    unsigned j0 = 0;
    for (; j0 + 32 <= N; j0 += 32) tile_f32_blockw<V16f, 16, 4, 2>(i0, j0, N, K, A, B, C, ldc, D, ldd);
    for (; j0 + 16 <= N; j0 += 16) tile_f32_blockw<V16f, 16, 4, 1>(i0, j0, N, K, A, B, C, ldc, D, ldd);
    for (; j0 + 8 <= N; j0 += 8) tile_f32_blockw<V8f_, 8, 4, 1>(i0, j0, N, K, A, B, C, ldc, D, ldd);
  }
  for (; i0 < M; ++i0) {
    unsigned j0 = 0;
    for (; j0 + 32 <= N; j0 += 32) tile_f32_blockw<V16f, 16, 1, 2>(i0, j0, N, K, A, B, C, ldc, D, ldd);
    for (; j0 + 16 <= N; j0 += 16) tile_f32_blockw<V16f, 16, 1, 1>(i0, j0, N, K, A, B, C, ldc, D, ldd);
    for (; j0 + 8 <= N; j0 += 8) tile_f32_blockw<V8f_, 8, 1, 1>(i0, j0, N, K, A, B, C, ldc, D, ldd);
  }
}

// ---- integers ----------------------------------------------------------------------------------------------------------------

// The wrapping multiply-add form, as the AVX2 kernel (kernels.inc), R rows by V vectors of L columns: every operand any 32-bit integer.
template <class VT, class VS, unsigned L, unsigned R, unsigned V>
inline void tile_i32_blockw(unsigned i0, unsigned j0, unsigned N, unsigned K, const int32_t* A, const int32_t* B, const int32_t* C, size_t ldc, unsigned block, bool sat,
                            int32_t* D, size_t ldd) {
  VT acc[R][V];
  for (unsigned r = 0; r < R; ++r)
    for (unsigned v = 0; v < V; ++v) {
      if (C) std::memcpy(&acc[r][v], C + size_t{i0 + r} * ldc + j0 + L * v, sizeof(VT));
      else acc[r][v] = VT{};
    }
  if (!sat) {
    for (unsigned k = 0; k < K; ++k) {
      VT b[V];
      for (unsigned v = 0; v < V; ++v) std::memcpy(&b[v], B + size_t{k} * N + j0 + L * v, sizeof(VT));
      for (unsigned r = 0; r < R; ++r)
        for (unsigned v = 0; v < V; ++v) acc[r][v] = acc[r][v] + static_cast<uint32_t>(A[size_t{i0 + r} * K + k]) * b[v];
    }
  } else {
    for (unsigned k0 = 0; k0 < K; k0 += block) {
      VT s[R][V] = {};
      const unsigned k1 = k0 + block < K ? k0 + block : K;
      for (unsigned k = k0; k < k1; ++k) {
        VT b[V];
        for (unsigned v = 0; v < V; ++v) std::memcpy(&b[v], B + size_t{k} * N + j0 + L * v, sizeof(VT));
        for (unsigned r = 0; r < R; ++r)
          for (unsigned v = 0; v < V; ++v) s[r][v] = s[r][v] + static_cast<uint32_t>(A[size_t{i0 + r} * K + k]) * b[v];
      }
      for (unsigned r = 0; r < R; ++r)
        for (unsigned v = 0; v < V; ++v) {
          // (acc + s clamped to the s32 range when it overflows: both of one sign and the sum of the other)
          const VT sum = acc[r][v] + s[r][v];
          const VS ov = reinterpret_cast<VS>((acc[r][v] ^ sum) & (s[r][v] ^ sum)) < 0;
          const VT neg = reinterpret_cast<VT>(reinterpret_cast<VS>(acc[r][v]) >> 31);
          const VT lim = (neg & 0x80000000u) | (~neg & 0x7fffffffu);
          const VT om = reinterpret_cast<VT>(ov);
          acc[r][v] = (om & lim) | (~om & sum);
        }
    }
  }
  for (unsigned r = 0; r < R; ++r)
    for (unsigned v = 0; v < V; ++v) std::memcpy(D + size_t{i0 + r} * ldd + j0 + L * v, &acc[r][v], sizeof(VT));
}

// The clamp of the running value plus a block's sum (kernels.inc's tile_i32_block), on 512 or 256 bits.
inline __m512i clamp_add512(__m512i acc, __m512i s) {
  const __m512i sum = _mm512_add_epi32(acc, s);
  const __mmask16 ov = _mm512_movepi32_mask(_mm512_and_si512(_mm512_xor_si512(acc, sum), _mm512_xor_si512(s, sum)));
  const __m512i lim = _mm512_mask_blend_epi32(_mm512_movepi32_mask(acc), _mm512_set1_epi32(0x7fffffff), _mm512_set1_epi32(static_cast<int>(0x80000000u)));
  return _mm512_mask_blend_epi32(ov, sum, lim);
}
inline __m256i clamp_add256(__m256i acc, __m256i s) {
  const __m256i sum = _mm256_add_epi32(acc, s);
  const __mmask8 ov = _mm256_movepi32_mask(_mm256_and_si256(_mm256_xor_si256(acc, sum), _mm256_xor_si256(s, sum)));
  const __m256i lim = _mm256_mask_blend_epi32(_mm256_movepi32_mask(acc), _mm256_set1_epi32(0x7fffffff), _mm256_set1_epi32(static_cast<int>(0x80000000u)));
  return _mm256_mask_blend_epi32(ov, sum, lim);
}

// The VNNI form: the operands as pairs of 16-bit numbers (A's pair (k, k + 1) of a row broadcast, B's of a column in a lane), each step one
// VPDPWSSD (acc += a0 * b0 + a1 * b1 on signed 16-bit halves, 32-bit wrapping) -- the same sum of exact products as the multiplier form's. A block
// (`block` products, even, or the whole K) is summed from zero and then added to the running value with the clamp, as the multiplier form does.
// Returns false, having written nothing, when an operand is not a 16-bit number (the caller then runs the multiplier form) or the shape is not this
// form's (an odd block under .sat).
struct VnniScratch {
  std::vector<int32_t> a, b;   // A as M x KP pairs; B as KP x N pairs (KP = K / 2 rounded up)
};

bool tile_i32_vnni(unsigned M, unsigned N, unsigned K, const int32_t* A, const int32_t* B, const int32_t* C, size_t ldc, unsigned block, bool sat, int32_t* D,
                   size_t ldd) {
  if (K == 0 || (sat && (block & 1) && block < K)) return false;
  const unsigned KP = (K + 1) / 2;
  thread_local VnniScratch sc;
  sc.a.resize(size_t{M} * KP);
  sc.b.resize(size_t{KP} * N);
  uint32_t bad = 0;   // some operand outside int16
  for (unsigned i = 0; i < M; ++i)
    for (unsigned kp = 0; kp < KP; ++kp) {
      const int32_t a0 = A[size_t{i} * K + 2 * kp], a1 = 2 * kp + 1 < K ? A[size_t{i} * K + 2 * kp + 1] : 0;
      bad |= static_cast<uint32_t>(a0 != static_cast<int16_t>(a0)) | static_cast<uint32_t>(a1 != static_cast<int16_t>(a1));
      sc.a[size_t{i} * KP + kp] = static_cast<int32_t>((static_cast<uint32_t>(a0) & 0xffffu) | (static_cast<uint32_t>(a1) << 16));
    }
  if (bad) return false;
  // B's pairs, and the check that each of its numbers is a signed 16-bit one (the value survives a round trip through 16 bits).
  const auto fits512 = [](__m512i x) { return _mm512_cmpneq_epi32_mask(_mm512_srai_epi32(_mm512_slli_epi32(x, 16), 16), x) == 0; };
  const auto fits256 = [](__m256i x) { return _mm256_cmpneq_epi32_mask(_mm256_srai_epi32(_mm256_slli_epi32(x, 16), 16), x) == 0; };
  unsigned j = 0;
  for (; j + 16 <= N; j += 16)
    for (unsigned kp = 0; kp < KP; ++kp) {
      const __m512i b0 = _mm512_loadu_si512(B + size_t{2 * kp} * N + j);
      const __m512i b1 = 2 * kp + 1 < K ? _mm512_loadu_si512(B + size_t{2 * kp + 1} * N + j) : _mm512_setzero_si512();
      if (!fits512(b0) || !fits512(b1)) return false;
      _mm512_storeu_si512(&sc.b[size_t{kp} * N + j], _mm512_or_si512(_mm512_and_si512(b0, _mm512_set1_epi32(0xffff)), _mm512_slli_epi32(b1, 16)));
    }
  for (; j + 8 <= N; j += 8)
    for (unsigned kp = 0; kp < KP; ++kp) {
      const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + size_t{2 * kp} * N + j));
      const __m256i b1 = 2 * kp + 1 < K ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + size_t{2 * kp + 1} * N + j)) : _mm256_setzero_si256();
      if (!fits256(b0) || !fits256(b1)) return false;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&sc.b[size_t{kp} * N + j]),
                          _mm256_or_si256(_mm256_and_si256(b0, _mm256_set1_epi32(0xffff)), _mm256_slli_epi32(b1, 16)));
    }
  // The products. Four rows at a time over 16 columns (two vectors of them where N allows), then the 8-column tail.
  const unsigned blk = sat ? block : K, blkp = (blk + 1) / 2;
  const auto rows = [&]<unsigned R>(std::integral_constant<unsigned, R>, unsigned i0) {
    unsigned jj = 0;
    for (; jj + 16 <= N; jj += 16) {
      __m512i acc[R];
      for (unsigned r = 0; r < R; ++r) acc[r] = C ? _mm512_loadu_si512(C + size_t{i0 + r} * ldc + jj) : _mm512_setzero_si512();
      for (unsigned kp0 = 0; kp0 < KP; kp0 += blkp) {
        const unsigned kp1 = kp0 + blkp < KP ? kp0 + blkp : KP;
        __m512i s[R];
        for (unsigned r = 0; r < R; ++r) s[r] = _mm512_setzero_si512();
        for (unsigned kp = kp0; kp < kp1; ++kp) {
          const __m512i b = _mm512_loadu_si512(&sc.b[size_t{kp} * N + jj]);
          for (unsigned r = 0; r < R; ++r) s[r] = _mm512_dpwssd_epi32(s[r], _mm512_set1_epi32(sc.a[size_t{i0 + r} * KP + kp]), b);
        }
        for (unsigned r = 0; r < R; ++r) acc[r] = sat ? clamp_add512(acc[r], s[r]) : _mm512_add_epi32(acc[r], s[r]);
      }
      for (unsigned r = 0; r < R; ++r) _mm512_storeu_si512(D + size_t{i0 + r} * ldd + jj, acc[r]);
    }
    for (; jj + 8 <= N; jj += 8) {
      __m256i acc[R];
      for (unsigned r = 0; r < R; ++r) acc[r] = C ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + size_t{i0 + r} * ldc + jj)) : _mm256_setzero_si256();
      for (unsigned kp0 = 0; kp0 < KP; kp0 += blkp) {
        const unsigned kp1 = kp0 + blkp < KP ? kp0 + blkp : KP;
        __m256i s[R];
        for (unsigned r = 0; r < R; ++r) s[r] = _mm256_setzero_si256();
        for (unsigned kp = kp0; kp < kp1; ++kp) {
          const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sc.b[size_t{kp} * N + jj]));
          for (unsigned r = 0; r < R; ++r) s[r] = _mm256_dpwssd_epi32(s[r], _mm256_set1_epi32(sc.a[size_t{i0 + r} * KP + kp]), b);
        }
        for (unsigned r = 0; r < R; ++r) acc[r] = sat ? clamp_add256(acc[r], s[r]) : _mm256_add_epi32(acc[r], s[r]);
      }
      for (unsigned r = 0; r < R; ++r) _mm256_storeu_si256(reinterpret_cast<__m256i*>(D + size_t{i0 + r} * ldd + jj), acc[r]);
    }
  };
  unsigned i0 = 0;
  for (; i0 + 4 <= M; i0 += 4) rows(std::integral_constant<unsigned, 4>{}, i0);
  for (; i0 < M; ++i0) rows(std::integral_constant<unsigned, 1>{}, i0);
  return true;
}

void tile_i32_k(unsigned M, unsigned N, unsigned K, const int32_t* A, const int32_t* B, const int32_t* C, size_t ldc, unsigned block, bool sat, int32_t* D,
                size_t ldd) {
  if (tile_i32_vnni(M, N, K, A, B, C, ldc, block, sat, D, ldd)) return;
  unsigned i0 = 0;
  for (; i0 + 4 <= M; i0 += 4) {
    unsigned j0 = 0;
    for (; j0 + 16 <= N; j0 += 16) tile_i32_blockw<V16u, V16i, 16, 4, 1>(i0, j0, N, K, A, B, C, ldc, block, sat, D, ldd);
    for (; j0 + 8 <= N; j0 += 8) tile_i32_blockw<V8u_, V8i_, 8, 4, 1>(i0, j0, N, K, A, B, C, ldc, block, sat, D, ldd);
  }
  for (; i0 < M; ++i0) {
    unsigned j0 = 0;
    for (; j0 + 16 <= N; j0 += 16) tile_i32_blockw<V16u, V16i, 16, 1, 1>(i0, j0, N, K, A, B, C, ldc, block, sat, D, ldd);
    for (; j0 + 8 <= N; j0 += 8) tile_i32_blockw<V8u_, V8i_, 8, 1, 1>(i0, j0, N, K, A, B, C, ldc, block, sat, D, ldd);
  }
}

// ---- binary64 ----------------------------------------------------------------------------------------------------------------

// D = C + A x B in binary64, k in order, each step one fused multiply-add in the host's current rounding mode (MXCSR, which the 512-bit form obeys
// as the scalar one does). Four rows at a time over eight columns.
void tile_f64_k(unsigned M, unsigned N, unsigned K, const double* A, const double* B, const double* C, double* D) {
  const auto rows = [&]<unsigned R>(std::integral_constant<unsigned, R>, unsigned i0) {
    unsigned j = 0;
    for (; j + 8 <= N; j += 8) {
      __m512d acc[R];
      for (unsigned r = 0; r < R; ++r) acc[r] = C ? _mm512_loadu_pd(C + size_t{i0 + r} * N + j) : _mm512_setzero_pd();
      for (unsigned k = 0; k < K; ++k) {
        const __m512d b = _mm512_loadu_pd(B + size_t{k} * N + j);
        for (unsigned r = 0; r < R; ++r) acc[r] = _mm512_fmadd_pd(_mm512_set1_pd(A[size_t{i0 + r} * K + k]), b, acc[r]);
      }
      for (unsigned r = 0; r < R; ++r) _mm512_storeu_pd(D + size_t{i0 + r} * N + j, acc[r]);
    }
    for (; j + 4 <= N; j += 4) {
      __m256d acc[R];
      for (unsigned r = 0; r < R; ++r) acc[r] = C ? _mm256_loadu_pd(C + size_t{i0 + r} * N + j) : _mm256_setzero_pd();
      for (unsigned k = 0; k < K; ++k) {
        const __m256d b = _mm256_loadu_pd(B + size_t{k} * N + j);
        for (unsigned r = 0; r < R; ++r) acc[r] = _mm256_fmadd_pd(_mm256_set1_pd(A[size_t{i0 + r} * K + k]), b, acc[r]);
      }
      for (unsigned r = 0; r < R; ++r) _mm256_storeu_pd(D + size_t{i0 + r} * N + j, acc[r]);
    }
    for (; j < N; ++j)
      for (unsigned r = 0; r < R; ++r) {
        double acc = C ? C[size_t{i0 + r} * N + j] : 0.0;
        for (unsigned k = 0; k < K; ++k) acc = __builtin_fma(A[size_t{i0 + r} * K + k], B[size_t{k} * N + j], acc);
        D[size_t{i0 + r} * N + j] = acc;
      }
  };
  unsigned i0 = 0;
  for (; i0 + 4 <= M; i0 += 4) rows(std::integral_constant<unsigned, 4>{}, i0);
  for (; i0 < M; ++i0) rows(std::integral_constant<unsigned, 1>{}, i0);
}
