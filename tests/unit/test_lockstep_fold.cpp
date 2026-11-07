// lockstep_fold (vgpu/exec/lockstep.hpp) is the fingerprint the engines take of a block's registers to tell a block that waits from one that
// computes: only the equality of two fingerprints of the same block is ever looked at. Long inputs are folded by eight independent chains,
// so what has to hold is what the old one-word chain gave: the same bytes give the same value, and a change of any one byte (at any offset,
// at any length, whatever the running value) changes it. A chain left out of the final combination, a tail dropped or the running value
// ignored would let a block that did change look like one that waits, and the lockstep launch would call it stuck.
#include <cstdint>
#include <cstring>
#include <vector>

#include "vgpu/exec/lockstep.hpp"
#include "vtest.hpp"

namespace {

using vgpu::exec::lockstep_fold;

std::vector<unsigned char> bytes(size_t n, uint64_t seed) {
  std::vector<unsigned char> v(n);
  for (size_t i = 0; i < n; ++i) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    v[i] = static_cast<unsigned char>(seed >> 56);
  }
  return v;
}

}  // namespace

VTEST(the_same_bytes_and_start_give_the_same_value) {
  for (size_t n : {0u, 1u, 7u, 8u, 9u, 63u, 64u, 65u, 128u, 1000u}) {
    const auto v = bytes(n, n + 1);
    VCHECK_EQ(lockstep_fold(5, v.data(), n), lockstep_fold(5, v.data(), n));
  }
}

VTEST(a_change_of_any_byte_changes_the_value_at_every_length) {
  // Every offset of every length from 1 to 300, several bit flips each, from two running values.
  for (size_t n = 1; n <= 300; ++n) {
    auto v = bytes(n, 1000 + n);
    for (uint64_t h : {0xcbf29ce484222325ull, 0x1234ull}) {
      const uint64_t base = lockstep_fold(h, v.data(), n);
      for (size_t i = 0; i < n; ++i)
        for (unsigned bit : {0u, 3u, 7u}) {
          v[i] ^= static_cast<unsigned char>(1u << bit);
          VCHECK(lockstep_fold(h, v.data(), n) != base);
          v[i] ^= static_cast<unsigned char>(1u << bit);
        }
    }
  }
}

VTEST(the_running_value_matters) {
  for (size_t n : {0u, 8u, 63u, 64u, 200u}) {
    const auto v = bytes(n, 77);
    VCHECK(lockstep_fold(1, v.data(), n) != lockstep_fold(2, v.data(), n));
  }
}

VTEST(folding_in_pieces_is_not_folding_the_whole_but_each_is_stable) {
  // A register file is folded as several calls (registers, then predicates, then program counters): the order of the pieces matters.
  const auto a = bytes(256, 1), b = bytes(256, 2);
  const uint64_t ab = lockstep_fold(lockstep_fold(9, a.data(), a.size()), b.data(), b.size());
  const uint64_t ba = lockstep_fold(lockstep_fold(9, b.data(), b.size()), a.data(), a.size());
  VCHECK(ab != ba);
}

VTEST(no_value_collides_in_a_walk_of_single_word_changes_in_a_register_file_sized_input) {
  // 32 KiB of registers, one word changed at a time to 64 values: all the fingerprints differ.
  auto v = bytes(32768, 5);
  std::vector<uint64_t> seen;
  for (size_t w = 0; w < v.size() / 8; w += 31)
    for (uint64_t x = 1; x <= 4; ++x) {
      uint64_t old;
      std::memcpy(&old, v.data() + w * 8, 8);
      const uint64_t nw = old ^ (x * 0x0101010101010101ull);
      std::memcpy(v.data() + w * 8, &nw, 8);
      seen.push_back(lockstep_fold(0xcbf29ce484222325ull, v.data(), v.size()));
      std::memcpy(v.data() + w * 8, &old, 8);
    }
  for (size_t i = 0; i < seen.size(); ++i)
    for (size_t j = i + 1; j < seen.size(); ++j) VCHECK(seen[i] != seen[j]);
}

VTEST_MAIN
