// The register-blocked kernels of host_gemm.hpp, included once per instruction
// set inside a namespace that sets kVB (the vector width in bytes) and, for
// the wider one, a target pragma. Nothing here may call a std:: template: a
// function compiled under the AVX2 pragma would share its mangled name with the
// generic instantiation, and the linker could keep either one.
//
// What is computed, for the accumulator type T and operands that a loader turns into T one element at a time:
//
//     acc(i, j) = ((0 + a(i,0)*b(0,j)) + a(i,1)*b(1,j)) + ...     (p = 0 .. k-1)
//
// each product rounded, then each sum rounded, in that order and with no fused
// multiply-add. That is what the loops this replaces computed, one output at a
// time; here sixteen or so outputs advance together, one per vector lane, but
// every lane runs its own chain in the same order. The chain is cut into
// blocks of KC terms; the partial sum waits in memory between blocks, which
// changes nothing because it is the same value the register held.

template <class T>
struct Lanes {
  typedef T type __attribute__((vector_size(kVB)));
  static constexpr int n = kVB / static_cast<int>(sizeof(T));
};

template <class T> struct IsFloat { static constexpr bool value = false; };
template <> struct IsFloat<float> { static constexpr bool value = true; };
template <class T> struct IsDouble { static constexpr bool value = false; };
template <> struct IsDouble<double> { static constexpr bool value = true; };

template <class T>
struct Ops {
  using V = typename Lanes<T>::type;
  static constexpr int L = Lanes<T>::n;
  static constexpr int MRV = 2;
  static constexpr int MR = MRV * L;
  static constexpr int NR = 4;
  static constexpr int MC = 128;  // a multiple of MR for every L (4, 8) and of the rows of a block
  static constexpr int NC = 128;  // a multiple of NR
  static constexpr int KC = 256;


#ifdef VGPU_HG_INTRIN
  // The same accumulation with the vectors written out: each accumulator takes  c = c + a * b  one rounded product and one rounded sum at a
  // time, in the order of the packed panels. (Intrinsics, because the compiler's own choice of registers for the broadcasts and
  // products varies with its version; this keeps the 12 accumulators in registers.)
  static inline __attribute__((always_inline)) void micro_intrin(const T* __restrict ap, const T* __restrict bp, int kc,
                                                                 T* __restrict acc, int ld) {
    if constexpr (IsFloat<T>::value) {
      __m256 c[NR][2];
      for (int j = 0; j < NR; ++j) {
        c[j][0] = _mm256_loadu_ps(acc + static_cast<size_t>(j) * ld);
        c[j][1] = _mm256_loadu_ps(acc + static_cast<size_t>(j) * ld + 8);
      }
      for (int p = 0; p < kc; ++p) {
        const __m256 a0 = _mm256_loadu_ps(ap + static_cast<size_t>(p) * MR);
        const __m256 a1 = _mm256_loadu_ps(ap + static_cast<size_t>(p) * MR + 8);
        const float* b = bp + static_cast<size_t>(p) * NR;
        for (int j = 0; j < NR; ++j) {
          const __m256 bj = _mm256_broadcast_ss(b + j);
          c[j][0] = _mm256_add_ps(c[j][0], _mm256_mul_ps(a0, bj));
          c[j][1] = _mm256_add_ps(c[j][1], _mm256_mul_ps(a1, bj));
        }
      }
      for (int j = 0; j < NR; ++j) {
        _mm256_storeu_ps(acc + static_cast<size_t>(j) * ld, c[j][0]);
        _mm256_storeu_ps(acc + static_cast<size_t>(j) * ld + 8, c[j][1]);
      }
    } else {
      __m256d c[NR][2];
      for (int j = 0; j < NR; ++j) {
        c[j][0] = _mm256_loadu_pd(acc + static_cast<size_t>(j) * ld);
        c[j][1] = _mm256_loadu_pd(acc + static_cast<size_t>(j) * ld + 4);
      }
      for (int p = 0; p < kc; ++p) {
        const __m256d a0 = _mm256_loadu_pd(ap + static_cast<size_t>(p) * MR);
        const __m256d a1 = _mm256_loadu_pd(ap + static_cast<size_t>(p) * MR + 4);
        const double* b = bp + static_cast<size_t>(p) * NR;
        for (int j = 0; j < NR; ++j) {
          const __m256d bj = _mm256_broadcast_sd(b + j);
          c[j][0] = _mm256_add_pd(c[j][0], _mm256_mul_pd(a0, bj));
          c[j][1] = _mm256_add_pd(c[j][1], _mm256_mul_pd(a1, bj));
        }
      }
      for (int j = 0; j < NR; ++j) {
        _mm256_storeu_pd(acc + static_cast<size_t>(j) * ld, c[j][0]);
        _mm256_storeu_pd(acc + static_cast<size_t>(j) * ld + 4, c[j][1]);
      }
    }
  }
#endif

  // One MR x NR tile: acc += a * b over kc terms, products and sums rounded
  // separately. ap is kc panels of MR, bp is kc panels of NR; acc has leading
  // dimension ld and is MR x NR.
  static inline __attribute__((always_inline)) void micro(const T* __restrict ap, const T* __restrict bp, int kc,
                                                          T* __restrict acc, int ld) {
#ifdef VGPU_HG_INTRIN
    if constexpr (MRV == 2 && (IsFloat<T>::value || IsDouble<T>::value)) {
      micro_intrin(ap, bp, kc, acc, ld);
      return;
    }
#endif
    V c[NR][MRV];
    for (int j = 0; j < NR; ++j)
      for (int v = 0; v < MRV; ++v) __builtin_memcpy(&c[j][v], acc + static_cast<size_t>(j) * ld + v * L, sizeof(V));
    for (int p = 0; p < kc; ++p) {
      V a[MRV];
      for (int v = 0; v < MRV; ++v) __builtin_memcpy(&a[v], ap + static_cast<size_t>(p) * MR + v * L, sizeof(V));
      for (int j = 0; j < NR; ++j) {
        const T bs = bp[static_cast<size_t>(p) * NR + j];
        V b;
        for (int l = 0; l < L; ++l) b[l] = bs;
        for (int v = 0; v < MRV; ++v) c[j][v] = c[j][v] + a[v] * b;
      }
    }
    for (int j = 0; j < NR; ++j)
      for (int v = 0; v < MRV; ++v) __builtin_memcpy(acc + static_cast<size_t>(j) * ld + v * L, &c[j][v], sizeof(V));
  }

  // op(A)(i0 .. i0+mr, p0 .. p0+kc) as panels of MR rows, p-major, zero beyond m. The elements are converted to T
  // here, a panel at a time, by the loader: ld(index into the stored matrix, i, p) -> value. (This is the only
  // place an operand is converted, so nothing the size of a matrix is ever made: a panel is at most MC x KC.)
  template <class LA>
  static void pack_a(bool ta, const LA& ld, int64_t lda, int64_t i0, int64_t mc, int64_t m, int64_t p0, int kc, T* out) {
    for (int64_t ir = 0; ir < mc; ir += MR) {
      T* dst = out + static_cast<size_t>(ir / MR) * MR * kc;
      // Rows of this panel that exist: the rest are zeros.
      int64_t rv = mc - ir < m - (i0 + ir) ? mc - ir : m - (i0 + ir);
      if (rv > MR) rv = MR;
      if (ta) {  // op(A)(i, p) = A[p + i*lda]: contiguous in p
        for (int r = 0; r < MR; ++r) {
          if (r >= rv) {
            for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * MR + r] = T(0);
            continue;
          }
          const int64_t i = i0 + ir + r, base = p0 + i * lda;
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * MR + r] = static_cast<T>(ld(base + p, i, p0 + p));
        }
      } else {  // A[i + p*lda]: contiguous in i, so p runs outside
        for (int p = 0; p < kc; ++p) {
          const int64_t base = i0 + ir + (p0 + p) * lda;
          T* const d = dst + static_cast<size_t>(p) * MR;
          for (int r = 0; r < rv; ++r) d[r] = static_cast<T>(ld(base + r, i0 + ir + r, p0 + p));
          for (int r = static_cast<int>(rv); r < MR; ++r) d[r] = T(0);
        }
      }
    }
  }

  // op(B)(p0 .. p0+kc, j0 .. j0+nc) as panels of NR columns, p-major, zero beyond n.
  template <class LB>
  static void pack_b(bool tb, const LB& ld, int64_t ldb, int64_t j0, int64_t nc, int64_t n, int64_t p0, int kc, T* out) {
    for (int64_t jr = 0; jr < nc; jr += NR) {
      T* dst = out + static_cast<size_t>(jr / NR) * NR * kc;
      int64_t cv = nc - jr < n - (j0 + jr) ? nc - jr : n - (j0 + jr);
      if (cv > NR) cv = NR;
      if (tb) {  // op(B)(p, j) = B[j + p*ldb]: contiguous in j, so p runs outside
        for (int p = 0; p < kc; ++p) {
          const int64_t base = j0 + jr + (p0 + p) * ldb;
          T* const d = dst + static_cast<size_t>(p) * NR;
          for (int c = 0; c < cv; ++c) d[c] = static_cast<T>(ld(base + c, j0 + jr + c, p0 + p));
          for (int c = static_cast<int>(cv); c < NR; ++c) d[c] = T(0);
        }
      } else {  // B[p + j*ldb]: contiguous in p
        for (int c = 0; c < NR; ++c) {
          if (c >= cv) {
            for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = T(0);
            continue;
          }
          const int64_t j = j0 + jr + c, base = p0 + j * ldb;
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = static_cast<T>(ld(base + p, j, p0 + p));
        }
      }
    }
  }

  // The sums of the block C(ic .. ic+mc, jc .. jc+nc) over all k, into acc
  // (leading dimension mcp = mc rounded up to MR, ncp likewise to NR).
  // apack holds mcp*KC, bpack ncp*KC elements.
  template <class LA, class LB>
  static void block(bool ta, bool tb, int64_t m, int64_t n, int64_t k, const LA& A, int64_t lda, const LB& B,
                    int64_t ldb, int64_t ic, int64_t mc, int64_t jc, int64_t nc, T* acc, T* apack, T* bpack) {
    const int64_t mcp = (mc + MR - 1) / MR * MR, ncp = (nc + NR - 1) / NR * NR;
    for (int64_t i = 0; i < mcp * ncp; ++i) acc[i] = T(0);
    for (int64_t pc = 0; pc < k; pc += KC) {
      const int kc = static_cast<int>(k - pc < KC ? k - pc : KC);
      pack_a(ta, A, lda, ic, mc, m, pc, kc, apack);
      pack_b(tb, B, ldb, jc, nc, n, pc, kc, bpack);
      for (int64_t jr = 0; jr < ncp; jr += NR)
        for (int64_t ir = 0; ir < mcp; ir += MR)
          micro(apack + static_cast<size_t>(ir / MR) * MR * kc, bpack + static_cast<size_t>(jr / NR) * NR * kc, kc,
                acc + ir + jr * mcp, static_cast<int>(mcp));
    }
  }
};
