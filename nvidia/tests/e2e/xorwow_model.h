// The host model of cuDNN's dropout generators (see dnn_dropout.cu): cuRAND's XORWOW, thread t seeded as
// curand_init(seed, t, 0) seeds it, element i of a tensor drawn from thread i % T and kept when
// curand_uniform of the output exceeds p.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace host {
struct Gf2 { uint32_t col[160][5]; };

inline void v_step(uint32_t v[5]) {
  const uint32_t t = v[0] ^ (v[0] >> 2);
  const uint32_t n4 = (v[4] ^ (v[4] << 4)) ^ (t ^ (t << 1));
  v[0] = v[1], v[1] = v[2], v[2] = v[3], v[3] = v[4], v[4] = n4;
}
inline void apply(const Gf2& m, const uint32_t in[5], uint32_t out[5]) {
  uint32_t r[5] = {0, 0, 0, 0, 0};
  for (int k = 0; k < 160; ++k)
    if (in[k / 32] >> (k % 32) & 1)
      for (int j = 0; j < 5; ++j) r[j] ^= m.col[k][j];
  for (int j = 0; j < 5; ++j) out[j] = r[j];
}
// The recurrence moved forward by 2^67 steps, which is where cuRAND starts each subsequence.
inline const Gf2& jump() {
  static Gf2 m;
  static bool built = false;
  if (!built) {
    for (int k = 0; k < 160; ++k) {
      uint32_t v[5] = {0, 0, 0, 0, 0};
      v[k / 32] = 1u << (k % 32);
      v_step(v);
      for (int j = 0; j < 5; ++j) m.col[k][j] = v[j];
    }
    for (int i = 0; i < 67; ++i) {
      Gf2 sq;
      for (int k = 0; k < 160; ++k) apply(m, m.col[k], sq.col[k]);
      m = sq;
    }
    built = true;
  }
  return m;
}
struct State { uint32_t d, v[5]; };
// States of threads 0..T-1 for a seed.
inline std::vector<State> seeded(unsigned long long seed, size_t T) {
  const uint32_t s0 = (uint32_t)seed ^ 0xaad26b49u, s1 = (uint32_t)(seed >> 32) ^ 0xf7dcefddu;
  const uint32_t t0 = 1099087573u * s0, t1 = 2591861531u * s1;
  State st;
  st.d = 6615241u + t1 + t0;
  st.v[0] = 123456789u + t0, st.v[1] = 362436069u ^ t0, st.v[2] = 521288629u + t1, st.v[3] = 88675123u ^ t1,
  st.v[4] = 5783321u + t0;
  std::vector<State> all(T);
  for (size_t t = 0; t < T; ++t) {
    all[t] = st;
    uint32_t nv[5];
    apply(jump(), st.v, nv);
    std::memcpy(st.v, nv, sizeof nv);
  }
  return all;
}
inline uint32_t next(State& s) {
  v_step(s.v);
  s.d += 362437u;
  return s.d + s.v[4];
}
inline float uniform(uint32_t x) {
  volatile float scaled = (float)x * 2.3283064365386963e-10f;
  return scaled + 1.1641532182693481e-10f;
}
// n decisions, element i from thread i % T; the states move on.
inline std::vector<uint8_t> draw(std::vector<State>& st, size_t n, float p) {
  std::vector<uint8_t> keep(n);
  for (size_t i = 0; i < n; ++i) keep[i] = uniform(next(st[i % st.size()])) > p;
  return keep;
}
}  // namespace host

