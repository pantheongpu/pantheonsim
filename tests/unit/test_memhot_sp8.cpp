// The memory manager's hot paths after the sp8 round, against what they stand for.
//
// load_fields reads a record's fields with one look at the chunk table when the record lies in one chunk, and field by field when it
// crosses a boundary; batch_window / load_batch / store_batch scan the addresses of a warp for "aligned and inside the allocation"
// and rule a host mapping out for the whole allocation at once, asking each address only where a host buffer is mapped in a device
// window. Each is held here to the answer the one-at-a-time calls give, at the places a shortcut breaks: a record across a chunk
// boundary, over a uniform and an untouched chunk, an access ending exactly at the allocation's end and one a byte past it, one
// below its start, and a host mapping inside the allocation.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
#include "vtest.hpp"

using namespace vgpu;

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

VTEST(load_fields_is_one_load_scalar_per_field_over_every_kind_of_chunk_and_across_boundaries) {
  Rng rng{0x5eed};
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(4 * kChunkSize);
  // chunk 0 real and random, chunk 1 uniform, chunk 2 real, chunk 3 untouched
  std::vector<uint8_t> bytes(kChunkSize);
  for (auto& b : bytes) b = static_cast<uint8_t>(rng.next());
  mm.write(p, bytes.data(), kChunkSize);
  const uint8_t pat = 0xAB;
  mm.fill(p + kChunkSize, &pat, 1, kChunkSize);
  for (auto& b : bytes) b = static_cast<uint8_t>(rng.next());
  mm.write(p + 2 * kChunkSize, bytes.data(), kChunkSize);
  static const uint8_t kinds[] = {0, 1, 2, 4, 8};
  unsigned taken = 0, crossing = 0, in_one = 0;
  for (int it = 0; it < 40000; ++it) {
    uint8_t sizes[4];
    const unsigned n = 1 + static_cast<unsigned>(rng.below(4));
    uint64_t total = 0;
    for (unsigned i = 0; i < n; ++i) total += (sizes[i] = kinds[rng.below(5)]);
    // Near a chunk boundary half the time, so records cross it.
    const uint64_t boundary = (1 + rng.below(3)) * kChunkSize;
    uint64_t off = it % 2 ? boundary - rng.below(16) : rng.below(4 * kChunkSize - 40);
    off &= ~uint64_t{3};   // records need not be aligned for every field: those load_fields declines
    if (it % 5 == 0) off &= ~uint64_t{7};
    if (off + total > 4 * kChunkSize) continue;
    uint64_t v[4] = {111, 111, 111, 111};
    const bool done = mm.load_fields(p + off, sizes, n, v);
    // The definition: each field one load_scalar of its width, end to end.
    uint64_t want[4];
    uint64_t at = off;
    bool aligned = true;
    for (unsigned i = 0; i < n; ++i) {
      if (sizes[i] == 0) {
        want[i] = 0;
        continue;
      }
      if (at & (sizes[i] - 1)) aligned = false;
      want[i] = aligned ? mm.load_scalar(p + at, sizes[i]) : 0;
      at += sizes[i];
    }
    if (!aligned) {
      VCHECK(!done);
      continue;
    }
    VCHECK(done);
    ++taken;
    (off % kChunkSize + total > kChunkSize ? crossing : in_one)++;
    for (unsigned i = 0; i < n; ++i)
      if (v[i] != want[i])
        throw vtest::Failure("load_fields field " + std::to_string(i) + " at offset " + std::to_string(off) + " differs");
  }
  VCHECK(taken > 5000);
  VCHECK(crossing > 200);
  VCHECK(in_one > 5000);
}

VTEST(a_batch_is_taken_exactly_where_every_access_is_aligned_and_inside_the_allocation) {
  MemoryManager mm(1ull << 28);
  const uint64_t size = 1000;   // not a multiple of the widest access
  const uint64_t p = mm.alloc(size);
  uint64_t out[4];
  const uint64_t vals[4] = {1, 2, 3, 4};
  for (uint32_t w : {1u, 2u, 4u, 8u}) {
    const uint64_t last = (size / w - 1) * w;   // the last access that fits
    const uint64_t ok[2] = {p, p + last};
    VCHECK(mm.load_batch(ok, 2, w, out));
    VCHECK(mm.store_batch(ok, 2, w, vals));
    // One whole element past the last that fits (where the allocation's end is not a multiple of w, a partial element is outside too).
    const uint64_t past[2] = {p, p + last + w};
    VCHECK(!mm.load_batch(past, 2, w, out));
    VCHECK(!mm.store_batch(past, 2, w, vals));
    const uint64_t below[3] = {p + w, p + 2 * w, p - w};
    VCHECK(!mm.load_batch(below, 3, w, out));
    VCHECK(!mm.store_batch(below, 3, w, vals));
    if (w > 1) {
      const uint64_t odd[2] = {p, p + w + 1};
      VCHECK(!mm.load_batch(odd, 2, w, out));
    }
  }
  // Exactly filling the allocation: 1000 bytes, 125 eight-byte words.
  const uint64_t last8[2] = {p, p + 992};
  VCHECK(mm.load_batch(last8, 2, 8, out));
  const uint64_t one_past8[2] = {p, p + 1000};
  VCHECK(!mm.load_batch(one_past8, 2, 8, out));
  const uint64_t one_past4[2] = {p, p + 1000};
  VCHECK(!mm.load_batch(one_past4, 2, 4, out));   // (1000 is aligned for 4 and is the first byte past)
  const uint64_t end4[2] = {p, p + 996};
  VCHECK(mm.load_batch(end4, 2, 4, out));
}

VTEST(a_batch_asks_each_address_where_a_host_buffer_is_mapped_inside_a_device_window) {
  MemoryManager mm(1ull << 28);
  const uint64_t p = mm.alloc(1 << 16);
  uint64_t out[3];
  const uint64_t vals[3] = {5, 6, 7};
  const uint64_t clear[3] = {p, p + 8, p + 8192};
  // Nothing mapped in a device window yet: the allocation as a whole is ruled clear.
  VCHECK(mm.load_batch(clear, 3, 4, out));
  std::vector<uint32_t> host(1024, 9);
  mm.map_host(p + 4096, host.data(), 4096);   // now one is, inside the allocation
  const uint64_t inside[3] = {p, p + 8, p + 4096 + 4};
  VCHECK(!mm.load_batch(inside, 3, 4, out));
  VCHECK(!mm.store_batch(inside, 3, 4, vals));
  const uint64_t first_byte[2] = {p + 4096, p};
  VCHECK(!mm.load_batch(first_byte, 2, 4, out));
  const uint64_t last_in[2] = {p, p + 8188};
  VCHECK(!mm.load_batch(last_in, 2, 4, out));
  const uint64_t after[2] = {p, p + 8192};
  VCHECK(mm.load_batch(after, 2, 4, out));
  const uint64_t before[2] = {p, p + 4092};
  VCHECK(mm.load_batch(before, 2, 4, out));
  // The same for the fields of a record: one inside the map declines, beside it does not.
  const uint8_t sizes[2] = {4, 4};
  uint64_t f[2];
  VCHECK(mm.load_fields(p + 4092, sizes, 2, f) == false);   // the second field starts in the map
  VCHECK(mm.load_fields(p + 4080, sizes, 2, f));
  VCHECK(mm.load_fields(p + 8192, sizes, 2, f) == true);
  mm.unmap_host(p + 4096);
}

VTEST(two_allocations_used_alternately_keep_their_answers_across_frees_of_others_and_of_either) {
  // A kernel that reads one allocation and writes another alternates its lookups between two; the thread's memory of lookups answers them from
  // its second way. It must stay right when an unrelated allocation is freed (the entries are revalidated, not dropped) and must stop
  // answering for an allocation that is freed.
  MemoryManager mm(1ull << 28);
  const uint64_t a = mm.alloc(4096), b = mm.alloc(4096), c = mm.alloc(4096);
  mm.store_scalar(a + 64, 4, 0xA1A1A1A1);
  mm.store_scalar(b + 64, 4, 0xB2B2B2B2);
  auto alternate = [&](bool a_alive) {
    for (int i = 0; i < 200; ++i) {
      if (a_alive) VCHECK_EQ(mm.load_scalar(a + 64, 4), 0xA1A1A1A1u);
      VCHECK_EQ(mm.load_scalar(b + 64, 4), 0xB2B2B2B2u);
      uint64_t out = 0;
      const uint64_t addrs[1] = {b + 64};
      VCHECK(mm.load_batch(addrs, 1, 4, &out));
      VCHECK_EQ(out, 0xB2B2B2B2u);
      if (a_alive) {
        const uint64_t aa[1] = {a + 64};
        VCHECK(mm.load_batch(aa, 1, 4, &out));
        VCHECK_EQ(out, 0xA1A1A1A1u);
      }
    }
  };
  alternate(true);
  mm.free(c);   // another allocation's free moves the table's epoch: the entries for a and b are still right
  alternate(true);
  mm.free(a);
  VCHECK_EQ(mm.load_scalar(b + 64, 4), 0xB2B2B2B2u);
  bool threw = false;
  try {
    mm.load_scalar(a + 64, 4);
  } catch (const Error&) {
    threw = true;
  }
  VCHECK(threw);
  uint64_t out = 0;
  const uint64_t aa[1] = {a + 64};
  VCHECK(!mm.load_batch(aa, 1, 4, &out));   // a batch declines: the freed one is the one-by-one calls' to report
  alternate(false);
}

VTEST_MAIN
