// The kernels, written once and compiled once per level: simd.cpp includes
// this file inside a namespace of the level's name, the second time under
// #pragma GCC target. Plain C++ and GCC vector types, so every level is the
// same source and the same bits; the intrinsics below are only for what the
// baseline cannot say (F16C), behind the macros the target pragma defines.
//
// Contraction is off for the whole file: a product that the card rounds
// before the sum must never be fused into an FMA by a build that has one.

namespace VGPU_SIMD_NS {

typedef float V8f __attribute__((vector_size(32)));

inline void f16_to_f32_k(const uint16_t* in, float* out, size_t n) {
  size_t i = 0;
#if defined(__F16C__) && defined(__AVX__)
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i))));
#endif
  for (; i < n; ++i) out[i] = f16_to_f32(in[i]);
}

inline void f32_to_f16_k(const float* in, uint16_t* out, size_t n) {
  size_t i = 0;
#if defined(__F16C__) && defined(__AVX__)
  for (; i + 8 <= n; i += 8)
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i),
                     _mm256_cvtps_ph(_mm256_loadu_ps(in + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#endif
  for (; i < n; ++i) out[i] = f32_to_f16_bits(in[i]);
}

// n 32-bit words of 16-bit (kind 0, 1) or tf32 (kind 2) operands, as floats.
inline void operands_to_f32(unsigned kind, const uint32_t* w, float* out, size_t n_words) {
  if (kind == 0) {
    uint16_t h[512];   // at most 4 registers * 32 lanes * 2 halves
    for (size_t i = 0; i < n_words; ++i) {
      h[2 * i] = static_cast<uint16_t>(w[i]);
      h[2 * i + 1] = static_cast<uint16_t>(w[i] >> 16);
    }
    f16_to_f32_k(h, out, 2 * n_words);
  } else if (kind == 1) {
    for (size_t i = 0; i < n_words; ++i) {
      out[2 * i] = bf16_to_f32(static_cast<uint16_t>(w[i]));
      out[2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(w[i] >> 16));
    }
  } else {
    // tf32: the top 19 bits of an f32; the rest are ignored.
    for (size_t i = 0; i < n_words; ++i) out[i] = std::bit_cast<float>(w[i] & 0xffffe000u);
  }
}

void hmma_k(unsigned kind, unsigned K, bool acc_f16, const uint32_t* a, const uint32_t* b, const uint32_t* c,
            uint32_t* out) {
  const unsigned per_reg = kind == 2 ? 1 : 2, L = 4 * per_reg;
  const unsigned a_regs = K / (2 * per_reg), b_regs = K / (4 * per_reg);
  alignas(32) float FA[256], FB[128];
  operands_to_f32(kind, a, FA, size_t{a_regs} * 32);
  operands_to_f32(kind, b, FB, size_t{b_regs} * 32);
  // The fragments are laid out so that, for a lane group g (lanes 4g..4g+3), the L consecutive elements
  // FA[r * 32 * per_reg + g * L ...] are one row segment of A: row g + 8 (r % 2), columns (r / 2) * L ...; and for
  // B they are one column segment: column g, rows r * L ... (PTX ISA 9.7.16).
  alignas(32) float A[16 * 16], Bm[16 * 8], Cm[16 * 8], Dm[16 * 8];
  for (unsigned r = 0; r < a_regs; ++r)
    for (unsigned g = 0; g < 8; ++g)
      for (unsigned j = 0; j < L; ++j) A[(g + 8 * (r & 1)) * K + (r >> 1) * L + j] = FA[r * 32 * per_reg + g * L + j];
  for (unsigned r = 0; r < b_regs; ++r)
    for (unsigned g = 0; g < 8; ++g)
      for (unsigned j = 0; j < L; ++j) Bm[(r * L + j) * 8 + g] = FB[r * 32 * per_reg + g * L + j];
  // C: element s of lane 4g+t is row g + 8 (s / 2), column 2t + s % 2.
  if (!c) {
    for (unsigned i = 0; i < 16 * 8; ++i) Cm[i] = 0.0f;
  } else if (!acc_f16) {
    for (unsigned s = 0; s < 4; ++s)
      for (unsigned l = 0; l < 32; ++l)
        Cm[(l / 4 + 8 * (s >> 1)) * 8 + 2 * (l % 4) + (s & 1)] = std::bit_cast<float>(c[s * 32 + l]);
  } else {
    float tmp[128];
    uint16_t h[128];
    for (unsigned i = 0; i < 64; ++i) {
      h[2 * i] = static_cast<uint16_t>(c[i]);
      h[2 * i + 1] = static_cast<uint16_t>(c[i] >> 16);
    }
    f16_to_f32_k(h, tmp, 128);   // tmp[(s / 2) * 64 + l * 2 + s % 2]
    for (unsigned s = 0; s < 4; ++s)
      for (unsigned l = 0; l < 32; ++l)
        Cm[(l / 4 + 8 * (s >> 1)) * 8 + 2 * (l % 4) + (s & 1)] = tmp[(s >> 1) * 64 + l * 2 + (s & 1)];
  }
  for (unsigned i0 = 0; i0 < 16; i0 += 4) {
    V8f acc0, acc1, acc2, acc3;
    std::memcpy(&acc0, &Cm[(i0 + 0) * 8], sizeof acc0);
    std::memcpy(&acc1, &Cm[(i0 + 1) * 8], sizeof acc1);
    std::memcpy(&acc2, &Cm[(i0 + 2) * 8], sizeof acc2);
    std::memcpy(&acc3, &Cm[(i0 + 3) * 8], sizeof acc3);
    for (unsigned k = 0; k < K; ++k) {
      V8f bv;
      std::memcpy(&bv, &Bm[k * 8], sizeof bv);
      const V8f p0 = A[(i0 + 0) * K + k] * bv, p1 = A[(i0 + 1) * K + k] * bv, p2 = A[(i0 + 2) * K + k] * bv,
                p3 = A[(i0 + 3) * K + k] * bv;
      acc0 = acc0 + p0;
      acc1 = acc1 + p1;
      acc2 = acc2 + p2;
      acc3 = acc3 + p3;
    }
    std::memcpy(&Dm[(i0 + 0) * 8], &acc0, sizeof acc0);
    std::memcpy(&Dm[(i0 + 1) * 8], &acc1, sizeof acc1);
    std::memcpy(&Dm[(i0 + 2) * 8], &acc2, sizeof acc2);
    std::memcpy(&Dm[(i0 + 3) * 8], &acc3, sizeof acc3);
  }
  if (!acc_f16) {
    for (unsigned s = 0; s < 4; ++s)
      for (unsigned l = 0; l < 32; ++l) {
        const float v = Dm[(l / 4 + 8 * (s >> 1)) * 8 + 2 * (l % 4) + (s & 1)];
        out[s * 32 + l] = v != v ? 0x7fffffffu : std::bit_cast<uint32_t>(v);
      }
  } else {
    float flat[128];   // flat[(s / 2) * 64 + l * 2 + s % 2], as above
    for (unsigned s = 0; s < 4; ++s)
      for (unsigned l = 0; l < 32; ++l)
        flat[(s >> 1) * 64 + l * 2 + (s & 1)] = Dm[(l / 4 + 8 * (s >> 1)) * 8 + 2 * (l % 4) + (s & 1)];
    uint16_t h[128];
    f32_to_f16_k(flat, h, 128);
    for (unsigned i = 0; i < 128; ++i)
      if (flat[i] != flat[i]) h[i] = 0x7fff;   // the canonical NaN, whatever the payload
    for (unsigned i = 0; i < 64; ++i) out[i] = h[2 * i] | (static_cast<uint32_t>(h[2 * i + 1]) << 16);
  }
}

}  // namespace VGPU_SIMD_NS
