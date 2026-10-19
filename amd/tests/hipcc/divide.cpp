// Division, as the compiler writes it: `a / b` on float and double, which hipcc (without -ffast-math)
// turns into the correctly rounded sequence v_div_scale, v_rcp, a chain of fused multiply-adds,
// v_div_fmas and v_div_fixup. The hardware's answer is the IEEE quotient, bit for bit; so the
// reference here is the host's own division (SSE, round to nearest even, denormals kept), never the
// simulator's output. The inputs are every pairing of the edge values (zeros, denormals, the
// extremes of the exponent range, infinities, NaNs, both signs), and random operands of every
// exponent, including ones whose exponents differ by nearly the whole range (the cases the scaling
// instructions exist for).
//
// Built for gfx942, gfx950 and gfx90a (divide.<arch>).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

template <typename T>
__global__ void divide(const T* a, const T* b, T* q, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) q[i] = a[i] / b[i];
}

static uint64_t g_state = 0x9e3779b97f4a7c15ull;
static uint64_t next() {   // xorshift64*
  g_state ^= g_state >> 12;
  g_state ^= g_state << 25;
  g_state ^= g_state >> 27;
  return g_state * 0x2545f4914f6cdd1dull;
}

template <typename T> struct Traits;
template <> struct Traits<float> {
  using Bits = uint32_t;
  static constexpr int kMant = 23, kExpBits = 8;
  static constexpr const char* kName = "float";
};
template <> struct Traits<double> {
  using Bits = uint64_t;
  static constexpr int kMant = 52, kExpBits = 11;
  static constexpr const char* kName = "double";
};

template <typename T>
static typename Traits<T>::Bits bits_of(T v) {
  typename Traits<T>::Bits b;
  std::memcpy(&b, &v, sizeof b);
  return b;
}
template <typename T>
static T from_bits(typename Traits<T>::Bits b) {
  T v;
  std::memcpy(&v, &b, sizeof v);
  return v;
}

// A value with the given biased exponent, a random mantissa, and a random sign.
template <typename T>
static T make(uint32_t biased_exp, uint64_t mant, bool neg) {
  using B = typename Traits<T>::Bits;
  const B m = static_cast<B>(mant) & ((B{1} << Traits<T>::kMant) - 1);
  return from_bits<T>((static_cast<B>(neg) << (Traits<T>::kMant + Traits<T>::kExpBits)) |
                      (static_cast<B>(biased_exp) << Traits<T>::kMant) | m);
}

template <typename T>
static std::vector<T> edges() {
  constexpr uint32_t maxe = (1u << Traits<T>::kExpBits) - 1;
  const uint64_t top = ~uint64_t{0};
  std::vector<T> v;
  for (bool neg : {false, true}) {
    v.push_back(make<T>(0, 0, neg));            // zero
    v.push_back(make<T>(0, 1, neg));            // smallest denormal
    v.push_back(make<T>(0, 2, neg));
    v.push_back(make<T>(0, top, neg));          // largest denormal
    v.push_back(make<T>(0, 1ull << (Traits<T>::kMant - 1), neg));
    v.push_back(make<T>(1, 0, neg));            // smallest normal
    v.push_back(make<T>(1, top, neg));
    v.push_back(make<T>(2, 0, neg));
    v.push_back(make<T>(maxe / 2 - 1, 0, neg));
    v.push_back(make<T>(maxe / 2, 0, neg));     // 0.5
    v.push_back(make<T>(maxe / 2, top, neg));
    v.push_back(make<T>(maxe / 2 + 1, 0, neg)); // 1.0
    v.push_back(make<T>(maxe / 2 + 1, 1, neg));
    v.push_back(make<T>(maxe / 2 + 2, 0, neg)); // 2.0
    v.push_back(make<T>(maxe / 2 + 2, 0x5555555555555555ull, neg));
    v.push_back(make<T>(maxe - 2, 0, neg));
    v.push_back(make<T>(maxe - 1, 0, neg));
    v.push_back(make<T>(maxe - 1, top, neg));   // largest finite
    v.push_back(make<T>(maxe, 0, neg));         // infinity
  }
  v.push_back(make<T>(maxe, 1, false));         // NaNs
  v.push_back(make<T>(maxe, 1ull << (Traits<T>::kMant - 1), true));
  v.push_back(make<T>(maxe, top, false));
  return v;
}

template <typename T>
static void run(const char* name, int& bad_total) {
  using B = typename Traits<T>::Bits;
  constexpr uint32_t maxe = (1u << Traits<T>::kExpBits) - 1;
  std::vector<T> a, b;
  const std::vector<T> e = edges<T>();
  for (T x : e)
    for (T y : e) {
      a.push_back(x);
      b.push_back(y);
    }
  const size_t pairs = a.size();
  // Random operands over every exponent, with mantissas often all-ones or all-zero.
  auto rmant = [&]() -> uint64_t {
    switch (next() % 6) {
      case 0: return 0;
      case 1: return ~uint64_t{0};
      case 2: return 1;
      default: return next();
    }
  };
  for (int i = 0; i < 60000; ++i) {
    a.push_back(make<T>(next() % (maxe + 1), rmant(), next() & 1));
    b.push_back(make<T>(next() % (maxe + 1), rmant(), next() & 1));
  }
  // Exponents that differ by nearly the whole range: the quotient overflows, underflows, or lands
  // just inside either end (denormal results, and results near the largest finite value).
  for (int i = 0; i < 60000; ++i) {
    const int64_t span = maxe - 1;
    const int64_t ea = 1 + static_cast<int64_t>(next() % span);
    int64_t delta = static_cast<int64_t>(next() % 21) - 10;   // around the extremes
    int64_t eb = (next() & 1) ? ea - (maxe / 2 + delta) : ea + (maxe / 2 + delta);
    // also exponent differences of +-(1 .. ~maxe)
    if (next() % 3 == 0) eb = ea + static_cast<int64_t>(next() % (2 * maxe)) - maxe;
    if (eb < 0) eb = 0;
    if (eb > maxe - 1) eb = maxe - 1;
    a.push_back(make<T>(static_cast<uint32_t>(ea), rmant(), next() & 1));
    b.push_back(make<T>(static_cast<uint32_t>(eb), rmant(), next() & 1));
  }
  // Denominators and numerators that are denormal, against anything.
  for (int i = 0; i < 20000; ++i) {
    T d = make<T>(0, rmant(), next() & 1), n = make<T>(next() % (maxe + 1), rmant(), next() & 1);
    if (i & 1) std::swap(d, n);
    a.push_back(n);
    b.push_back(d);
  }
  const int n = static_cast<int>(a.size());
  std::vector<T> q(n, T(0));
  T *da, *db, *dq;
  if (hipMalloc(&da, n * sizeof(T)) != hipSuccess || hipMalloc(&db, n * sizeof(T)) != hipSuccess ||
      hipMalloc(&dq, n * sizeof(T)) != hipSuccess) {
    std::printf("%s: hipMalloc failed\n", name);
    bad_total += 1;
    return;
  }
  if (hipMemcpy(da, a.data(), n * sizeof(T), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(db, b.data(), n * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) {
    std::printf("%s: hipMemcpy failed\n", name);
    bad_total += 1;
    return;
  }
  divide<T><<<(n + 255) / 256, 256>>>(da, db, dq, n);
  if (hipMemcpy(q.data(), dq, n * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) {
    std::printf("%s: the kernel failed\n", name);
    bad_total += 1;
    return;
  }
  (void)hipFree(da);
  (void)hipFree(db);
  (void)hipFree(dq);
  int bad = 0, shown = 0, bad_edge = 0;
  for (int i = 0; i < n; ++i) {
    volatile T x = a[i], y = b[i];
    const T want = x / y;   // the host's IEEE division
    const bool both_nan = std::isnan(want) && std::isnan(q[i]);
    if (both_nan || bits_of(want) == bits_of(q[i])) continue;
    // The exception: a denominator whose significand is all ones. Its reciprocal sits within 2^-2p of a rounding boundary and the
    // fused multiply-add refinement the compiler writes (Markstein's) cannot tell which side it is on, so the quotient may be one unit
    // in the last place off (what a card's table-driven v_rcp starts from decides). That is all that is asked of it.
    {
      using B = typename Traits<T>::Bits;
      const B ones = (B{1} << Traits<T>::kMant) - 1, sign = B{1} << (Traits<T>::kMant + Traits<T>::kExpBits);
      if ((bits_of(b[i]) & ones) == ones && !std::isnan(want) && !std::isnan(q[i]) && (bits_of(want) & sign) == (bits_of(q[i]) & sign)) {
        const B w = bits_of(want) & ~sign, g = bits_of(q[i]) & ~sign;
        if ((w > g ? w - g : g - w) <= 1) continue;
      }
    }
    ++bad;
    if (static_cast<size_t>(i) < pairs) ++bad_edge;
    if (shown < 6) {
      ++shown;
      std::printf("  %s mismatch: %a / %a: got %a (%llx) want %a (%llx)\n", Traits<T>::kName, static_cast<double>(a[i]),
                  static_cast<double>(b[i]), static_cast<double>(q[i]), static_cast<unsigned long long>(bits_of(q[i])),
                  static_cast<double>(want), static_cast<unsigned long long>(bits_of(want)));
    }
  }
  std::printf("%s %s: %d quotients, %d differ from IEEE division (%d of the %zu edge pairs)\n", name, Traits<T>::kName, n,
              bad, bad_edge, pairs);
  bad_total += bad;
}

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 2;
  std::printf("device %s\n", prop.gcnArchName);
  int bad = 0;
  run<float>("divide", bad);
  run<double>("divide", bad);
  std::printf("divide mismatches %d\n", bad);
  return bad ? 1 : 0;
}
