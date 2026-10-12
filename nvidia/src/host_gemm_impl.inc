// The register-blocked kernels of host_gemm.hpp, included once per instruction
// set inside a namespace that sets kVB (the vector width in bytes) and, for
// the wider one, a target pragma. Nothing here may call a std:: template: a
// function compiled under the AVX2 pragma would share its mangled name with the
// generic instantiation, and the linker could keep either one.
//
// What is computed, for the operand types T (accumulator) and S (stored):
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

template <class T, class S>
struct Ops {
  using V = typename Lanes<T>::type;
  static constexpr int L = Lanes<T>::n;
  static constexpr int MRV = 2;
  static constexpr int MR = MRV * L;
  static constexpr int NR = 6;
  static constexpr int MC = 128;  // a multiple of MR for every L (4, 8) and of the rows of a block
  static constexpr int NC = 192;  // a multiple of NR
  static constexpr int KC = 256;

  static inline T conv(S s) { return static_cast<T>(s); }

  // One MR x NR tile: acc += a * b over kc terms, products and sums rounded
  // separately. ap is kc panels of MR, bp is kc panels of NR; acc has leading
  // dimension ld and is MR x NR.
  static inline __attribute__((always_inline)) void micro(const T* __restrict ap, const T* __restrict bp, int kc,
                                                          T* __restrict acc, int ld) {
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

  // op(A)(i0 .. i0+mr, p0 .. p0+kc) as panels of MR rows, p-major, zero beyond m.
  static void pack_a(bool ta, const S* A, int64_t lda, int64_t i0, int64_t mc, int64_t m, int64_t p0, int kc, T* out) {
    for (int64_t ir = 0; ir < mc; ir += MR) {
      T* dst = out + static_cast<size_t>(ir / MR) * MR * kc;
      for (int r = 0; r < MR; ++r) {
        const int64_t i = i0 + ir + r;
        if (ir + r >= mc || i >= m) {
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * MR + r] = T(0);
        } else if (ta) {  // op(A)(i, p) = A[p + i*lda]: contiguous in p
          const S* src = A + p0 + i * lda;
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * MR + r] = conv(src[p]);
        } else {  // A[i + p*lda]
          const S* src = A + i + p0 * lda;
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * MR + r] = conv(src[static_cast<size_t>(p) * lda]);
        }
      }
    }
  }

  // op(B)(p0 .. p0+kc, j0 .. j0+nc) as panels of NR columns, p-major, zero beyond n.
  static void pack_b(bool tb, const S* B, int64_t ldb, int64_t j0, int64_t nc, int64_t n, int64_t p0, int kc, T* out) {
    for (int64_t jr = 0; jr < nc; jr += NR) {
      T* dst = out + static_cast<size_t>(jr / NR) * NR * kc;
      for (int c = 0; c < NR; ++c) {
        const int64_t j = j0 + jr + c;
        if (jr + c >= nc || j >= n) {
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = T(0);
        } else if (tb) {  // op(B)(p, j) = B[j + p*ldb]
          const S* src = B + j + p0 * ldb;
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = conv(src[static_cast<size_t>(p) * ldb]);
        } else {  // B[p + j*ldb]: contiguous in p
          const S* src = B + p0 + j * ldb;
          for (int p = 0; p < kc; ++p) dst[static_cast<size_t>(p) * NR + c] = conv(src[p]);
        }
      }
    }
  }

  // The sums of the block C(ic .. ic+mc, jc .. jc+nc) over all k, into acc
  // (leading dimension mcp = mc rounded up to MR, ncp likewise to NR).
  // apack holds mcp*KC, bpack ncp*KC elements.
  static void block(bool ta, bool tb, int64_t m, int64_t n, int64_t k, const S* A, int64_t lda, const S* B,
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
