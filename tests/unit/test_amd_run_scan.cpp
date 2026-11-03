// The run length of a wave's addresses (include/vgpu/exec/amd_run_scan.hpp, used by Machine::memory_run) against the definition, the lane
// by lane loop: rows of every kind -- a perfect run of every length, a run broken at every lane, a high word that differs in one lane,
// a low word that wraps round 2^32 inside the run (the vector pass hands those to the loop), the base form (one row), strides of 4, 8
// and 16, a first lane that is not the wave's first, a reach that is not a multiple of four.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "vgpu/exec/amd_run_scan.hpp"
#include "vtest.hpp"

namespace {
struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};
}  // namespace

VTEST(the_vector_run_length_is_the_loops_for_every_kind_of_row) {
  using namespace vgpu::exec::amdrun;
  Rng rng{0x5eed1234abcdULL};
  unsigned broke = 0, whole = 0, wrapped = 0, pair_form = 0, base_form = 0;
  for (int it = 0; it < 200000; ++it) {
    const uint32_t stride = 4u << rng.below(3);
    const uint32_t reach = 1 + static_cast<uint32_t>(rng.below(64));
    const bool pair = rng.below(3) != 0;
    // The first lane's address: mostly well away from a word's wrap, sometimes right at it.
    uint64_t first = pair ? rng.next() & 0x00007FFFFFFFFFFCull : rng.next() & 0xFFFFFFFCull;
    if (rng.below(6) == 0) first = (first & ~uint64_t{0xFFFFFFFF}) | ((0xFFFFFFFFull - 4 * rng.below(64)) & ~uint64_t{3});
    if (!pair && rng.below(6) == 0) first = (0xFFFFFFFFull - 4 * rng.below(64)) & ~uint64_t{3};
    std::vector<uint32_t> lo(64 + 8), hi(64 + 8);
    for (uint32_t k = 0; k < 64 + 8; ++k) {
      const uint64_t a = first + uint64_t{k} * stride;
      lo[k] = static_cast<uint32_t>(a);
      hi[k] = static_cast<uint32_t>(a >> 32);
    }
    // Break the run: a different low word, a different high word, or nothing.
    const unsigned how = static_cast<unsigned>(rng.below(5));
    const uint32_t at = static_cast<uint32_t>(rng.below(reach));
    if (how == 0) lo[at] += 4 * (1 + static_cast<uint32_t>(rng.below(3)));
    else if (how == 1 && pair) hi[at] ^= 1u << rng.below(32);
    else if (how == 2) lo[at] ^= 1u << rng.below(32);
    else if (how == 3 && reach > 1) {   // two lanes swapped
      const uint32_t b = static_cast<uint32_t>(rng.below(reach));
      std::swap(lo[at], lo[b]);
      if (pair) std::swap(hi[at], hi[b]);
    }
    const uint32_t* h = pair ? hi.data() : nullptr;
    // The pair form is given the lane's address less an offset the caller took off again; the base form lo[0].
    const uint64_t start = pair ? (static_cast<uint64_t>(hi[0]) << 32 | lo[0]) : uint64_t{lo[0]};
    const uint32_t want = run_scan_reference(lo.data(), h, reach, start, stride);
    const uint32_t got = run_scan(lo.data(), h, reach, start, stride);
    if (got != want)
      throw vtest::Failure("run length " + std::to_string(got) + ", the loop says " + std::to_string(want) + " (reach " + std::to_string(reach) +
                           ", stride " + std::to_string(stride) + (pair ? ", pair form" : ", base form") + ")");
    broke += want < reach;
    whole += want == reach && reach > 8;
    wrapped += (pair ? (start & 0xFFFFFFFFu) : start) + uint64_t{reach - 1} * stride >= (uint64_t{1} << 32);
    (pair ? pair_form : base_form) += 1;
  }
  VCHECK(broke > 20000);
  VCHECK(whole > 10000);
  VCHECK(wrapped > 3000);
  VCHECK(pair_form > 100000 && base_form > 40000);
}

// Whether any two neighbouring addresses of a list are equal (Machine::wave_scalars decides from it whether the pass that copies a repeated
// address's value runs) against the definition: lists of every length, with one pair equal at every position (the two halves of an
// address separately equal too: a pair that differs in the low or the high word only is not equal), and with none.
VTEST(a_pair_of_equal_neighbours_is_found_wherever_it_is) {
  using namespace vgpu::exec::amdrun;
  Rng rng{0xfeedbeef1234ULL};
  unsigned found = 0, none = 0;
  for (int it = 0; it < 200000; ++it) {
    const uint32_t n = static_cast<uint32_t>(rng.below(66));
    std::vector<uint64_t> a(n + 4);
    for (auto& v : a) v = rng.next();
    // Neighbours that agree in one word but not the other, and a few repeats.
    for (uint32_t k = 1; k < n; ++k) {
      const unsigned how = static_cast<unsigned>(rng.below(24));
      if (how == 0) a[k] = a[k - 1];
      else if (how == 1) a[k] = (a[k - 1] & 0xFFFFFFFFull) | (a[k] & ~0xFFFFFFFFull);
      else if (how == 2) a[k] = (a[k - 1] & ~0xFFFFFFFFull) | (a[k] & 0xFFFFFFFFull);
    }
    // (Entries past n are never looked at: make them equal to the last one.)
    for (uint32_t k = n; k < n + 4; ++k) a[k] = n ? a[n - 1] : 0;
    const bool want = any_adjacent_equal_reference(a.data(), n), got = any_adjacent_equal(a.data(), n);
    if (got != want)
      throw vtest::Failure(std::string("any_adjacent_equal says ") + (got ? "yes" : "no") + ", the loop says " + (want ? "yes" : "no") + " (n " +
                           std::to_string(n) + ")");
    found += want, none += !want;
  }
  VCHECK(found > 20000 && none > 5000);
  // Exactly one equal pair, at every position and every length.
  for (uint32_t n = 0; n <= 65; ++n)
    for (uint32_t at = 1; at < n; ++at) {
      std::vector<uint64_t> a(n);
      for (uint32_t k = 0; k < n; ++k) a[k] = 0x1000000001ull * (k + 1);
      VCHECK(!any_adjacent_equal(a.data(), n));
      a[at] = a[at - 1];
      VCHECK(any_adjacent_equal(a.data(), n));
    }
}

// Whether every address equals the first (Machine::wave_scalars reads such a load once) against the definition: every length, a list that
// differs from the first at one position only (in the low word, in the high word), and lists that are all one value.
VTEST(a_list_of_one_address_is_told_from_one_that_differs_anywhere) {
  using namespace vgpu::exec::amdrun;
  Rng rng{0x1234abcd5678ULL};
  for (uint32_t n = 0; n <= 66; ++n) {
    for (uint32_t at = 0; at <= n; ++at) {
      for (unsigned how = 0; how < 3; ++how) {
        std::vector<uint64_t> a(n, 0x00007F1200345670ull);
        if (at < n) a[at] ^= how == 0 ? 4 : how == 1 ? (uint64_t{1} << 40) : (uint64_t{1} << 63);
        VCHECK(all_equal(a.data(), n) == all_equal_reference(a.data(), n));
        VCHECK(all_equal(a.data(), n) == (at >= n || n == 1));   // (the list is one value, or one entry was changed: that is a difference unless it is the only one)
      }
    }
  }
  for (int it = 0; it < 100000; ++it) {
    const uint32_t n = static_cast<uint32_t>(rng.below(66));
    std::vector<uint64_t> a(n, rng.next());
    if (rng.below(2) && n) a[rng.below(n)] = rng.next();
    VCHECK(all_equal(a.data(), n) == all_equal_reference(a.data(), n));
  }
}

VTEST_MAIN
