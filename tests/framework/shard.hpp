// Which part of a test this process runs, for the long tests that CMakeLists.txt registers as parallel shards (vgpu_shard_tests, docs/ci.md).
//
//   const vtest::Shard shard("VGPU_GATHER_SHARD", "VGPU_GATHER_LIST");
//   ... for each item, numbered 0, 1, 2... the same way whatever the shard: if (!shard.mine(item)) continue;
//       if (shard.list) { std::printf("%s\n", name_of_item); continue; } ...
//
// VGPU_..._SHARD=k/n (0 <= k < n) makes the process take every item whose number is k modulo n; unset, it takes all of them. VGPU_..._LIST=1
// makes it print the items it would take, one a line, and run nothing (tests/lint/check_shard_partition.sh proves the shards together are the whole).
// A shard must keep every item's seed and numbering what they are in the whole run, so the n shards run exactly the runs of the unsharded test.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

#include "vtest.hpp"

namespace vtest {

struct Shard {
  uint64_t k = 0, n = 1;
  bool list = false;
  Shard(const char* var, const char* list_var) {
    if (const char* sh = std::getenv(var)) {
      char* end = nullptr;
      k = std::strtoull(sh, &end, 10);
      if (*end != '/') throw Failure(std::string(var) + " wants k/n, got '" + sh + "'");
      n = std::strtoull(end + 1, &end, 10);
      if (*end || n == 0 || k >= n) throw Failure(std::string(var) + " wants k/n with k < n, got '" + sh + "'");
    }
    list = std::getenv(list_var) != nullptr;
  }
  bool mine(uint64_t item) const { return item % n == k; }
  // A "the test ran enough" bound for the whole test, for a shard that runs about 1/n of it (and a margin for the uneven split).
  uint64_t at_least(uint64_t whole) const { return n == 1 ? whole : whole / (2 * n); }
};

}  // namespace vtest
