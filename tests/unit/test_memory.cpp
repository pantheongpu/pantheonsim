// Unit tests for the virtual device memory manager.
#include "vgpu/memory.hpp"

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "vgpu/error.hpp"
#include "vtest.hpp"

using vgpu::Err;
using vgpu::Error;
using vgpu::MemoryManager;

VTEST(alloc_write_read_roundtrip) {
  MemoryManager mm(1 << 20);
  uint64_t p = mm.alloc(1024);
  VCHECK(p >= vgpu::kDeviceVaBase);
  std::vector<uint8_t> in(1024), out(1024, 0xEE);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 7);
  mm.write(p, in.data(), in.size());
  mm.read(p, out.data(), out.size());
  VCHECK(in == out);
  VCHECK_EQ(mm.used(), 1024ull);
  mm.free(p);
  VCHECK_EQ(mm.used(), 0ull);
}

VTEST(untouched_memory_reads_zero) {
  MemoryManager mm(1 << 20);
  uint64_t p = mm.alloc(4096);
  std::vector<uint8_t> out(4096, 0xAB);
  mm.read(p, out.data(), out.size());
  for (uint8_t b : out) VCHECK_EQ(int(b), 0);
}

VTEST(huge_virtual_alloc_is_cheap) {
  // A virtual H200 claims 141 GB; the host must not need it. Allocate 100 GiB,
  // touch a few spots far apart, verify sparse behavior.
  MemoryManager mm(151397597184ull);
  uint64_t p = mm.alloc(100ull << 30);
  uint64_t marker = 0xDEADBEEFCAFEF00Dull;
  mm.store_scalar(p, 8, marker);
  mm.store_scalar(p + (99ull << 30), 8, marker);
  VCHECK_EQ(mm.load_scalar(p, 8), marker);
  VCHECK_EQ(mm.load_scalar(p + (99ull << 30), 8), marker);
  VCHECK_EQ(mm.load_scalar(p + (50ull << 30), 8), 0ull);
  mm.free(p);
}

VTEST(oom_reports_usage) {
  MemoryManager mm(1000);
  (void)mm.alloc(800);
  auto err = VCAPTURE(Error, mm.alloc(300));
  VCHECK(err.code() == Err::OutOfMemory);
  VCHECK_CONTAINS(err.what(), "requested 300");
  VCHECK_CONTAINS(err.what(), "800");
}

VTEST(zero_byte_alloc_rejected) {
  MemoryManager mm(1000);
  auto err = VCAPTURE(Error, mm.alloc(0));
  VCHECK(err.code() == Err::InvalidValue);
}

VTEST(oob_write_detected_with_context) {
  MemoryManager mm(1 << 20);
  uint64_t p = mm.alloc(256);
  std::vector<uint8_t> buf(512);
  auto err = VCAPTURE(Error, mm.write(p + 128, buf.data(), buf.size()));
  VCHECK(err.code() == Err::OutOfBounds);
  VCHECK_CONTAINS(err.what(), "256-byte allocation");
}

VTEST(invalid_pointer_detected) {
  MemoryManager mm(1 << 20);
  uint8_t b = 0;
  auto err = VCAPTURE(Error, mm.read(vgpu::kDeviceVaBase + 0x123456, &b, 1));
  VCHECK(err.code() == Err::InvalidPointer);
}

VTEST(host_pointer_hint) {
  MemoryManager mm(1 << 20);
  uint8_t b = 0;
  auto err = VCAPTURE(Error, mm.read(0x1000, &b, 1));
  VCHECK(err.code() == Err::InvalidPointer);
  VCHECK_CONTAINS(err.what(), "host pointer");
}

VTEST(double_free_detected) {
  MemoryManager mm(1 << 20);
  uint64_t p = mm.alloc(64);
  mm.free(p);
  auto err = VCAPTURE(Error, mm.free(p));
  VCHECK(err.code() == Err::DoubleFree);
}

VTEST(interior_free_detected) {
  MemoryManager mm(1 << 20);
  uint64_t p = mm.alloc(1024);
  auto err = VCAPTURE(Error, mm.free(p + 8));
  VCHECK(err.code() == Err::InvalidFree);
  VCHECK_CONTAINS(err.what(), "interior");
}

VTEST(use_after_free_detected) {
  MemoryManager mm(1 << 20);
  uint64_t p = mm.alloc(64);
  mm.free(p);
  uint8_t b = 0;
  auto err = VCAPTURE(Error, mm.read(p, &b, 1));
  VCHECK(err.code() == Err::UseAfterFree);
}

VTEST(freed_vas_never_reused) {
  MemoryManager mm(1 << 20);
  uint64_t p1 = mm.alloc(64);
  mm.free(p1);
  uint64_t p2 = mm.alloc(64);
  VCHECK(p1 != p2);
}

VTEST(misaligned_scalar_access_detected) {
  MemoryManager mm(1 << 20);
  uint64_t p = mm.alloc(64);
  auto err = VCAPTURE(Error, mm.load_scalar(p + 3, 4));
  VCHECK(err.code() == Err::MisalignedAccess);
}

VTEST(write_spanning_chunks) {
  MemoryManager mm(1 << 30);
  uint64_t p = mm.alloc(3 * vgpu::kChunkSize);
  std::vector<uint8_t> in(2 * vgpu::kChunkSize);  // straddles two chunk boundaries, stays in bounds
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i);
  uint64_t at = p + vgpu::kChunkSize - 50;  // straddles chunk boundaries
  mm.write(at, in.data(), in.size());
  std::vector<uint8_t> out(in.size());
  mm.read(at, out.data(), out.size());
  VCHECK(in == out);
}

VTEST_MAIN

// --- fill(): memset without a host staging buffer -------------------------
// A memset used to build a host copy the size of the range and then write it
// in, so setting N bytes of device memory cost N bytes of chunks plus N bytes
// of staging, live at the same time. These pin the behaviour that replaced it.

namespace {
// Resident set size in bytes, from the second field of /proc/self/statm.
uint64_t resident_bytes() {
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  unsigned long total = 0, resident = 0;
  const int got = std::fscanf(f, "%lu %lu", &total, &resident);
  std::fclose(f);
  if (got != 2) return 0;
  return static_cast<uint64_t>(resident) * static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
}
}  // namespace

VTEST(fill_byte_pattern_spans_chunks) {
  MemoryManager mm(1 << 20);
  const uint64_t n = vgpu::kChunkSize * 3 + 77;  // deliberately not chunk-aligned
  uint64_t p = mm.alloc(n);
  const uint8_t v = 0xAB;
  mm.fill(p, &v, 1, n);
  std::vector<uint8_t> out(n, 0);
  mm.read(p, out.data(), n);
  for (uint64_t i = 0; i < n; ++i) VCHECK_EQ(out[i], 0xAB);
  mm.free(p);
}

VTEST(fill_repeats_four_byte_pattern_in_phase) {
  // cuMemsetD32 repeats a 4-byte value; the phase must survive the chunk
  // boundary the fill crosses, not restart at each chunk.
  MemoryManager mm(1 << 20);
  const uint64_t n = vgpu::kChunkSize + 4 * 5;
  uint64_t p = mm.alloc(n);
  const uint32_t word = 0x11223344u;
  mm.fill(p, reinterpret_cast<const uint8_t*>(&word), 4, n);
  std::vector<uint8_t> out(n, 0);
  mm.read(p, out.data(), n);
  const uint8_t* w = reinterpret_cast<const uint8_t*>(&word);
  for (uint64_t i = 0; i < n; ++i) VCHECK_EQ(out[i], w[i % 4]);
  mm.free(p);
}

VTEST(fill_partial_range_leaves_neighbours_alone) {
  MemoryManager mm(1 << 20);
  const uint64_t n = vgpu::kChunkSize * 2;
  uint64_t p = mm.alloc(n);
  std::vector<uint8_t> seed(n, 0x5A);
  mm.write(p, seed.data(), n);
  const uint8_t v = 0xFF;
  mm.fill(p + 100, &v, 1, 50);
  std::vector<uint8_t> out(n, 0);
  mm.read(p, out.data(), n);
  for (uint64_t i = 0; i < n; ++i)
    VCHECK_EQ(out[i], (i >= 100 && i < 150) ? 0xFF : 0x5A);
  mm.free(p);
}

VTEST(zero_fill_does_not_materialize_chunks) {
  // Zeroing memory nothing has touched is already true of that memory, so it
  // must not cost host RAM. This is what keeps a workload that allocates most
  // of a large device and zeroes it from pulling the whole device into RSS.
  MemoryManager mm(1ull << 40);
  const uint64_t n = 256ull * 1024 * 1024;
  uint64_t p = mm.alloc(n);
  const uint64_t before = resident_bytes();
  const uint8_t zero = 0;
  mm.fill(p, &zero, 1, n);
  const uint64_t after = resident_bytes();
  // Allow slack for unrelated allocator noise, but nothing near the 256 MiB
  // that materializing every chunk would cost.
  VCHECK(after < before + (32ull * 1024 * 1024));
  // It still reads back as zero.
  std::vector<uint8_t> out(4096, 0xEE);
  mm.read(p + n - 4096, out.data(), out.size());
  for (uint8_t b : out) VCHECK_EQ(b, 0);
  mm.free(p);
}

VTEST(nonzero_fill_costs_about_one_copy) {
  // The staging buffer made a fill of N bytes cost about 2N. Hold it to
  // roughly one copy.
  MemoryManager mm(1ull << 40);
  const uint64_t n = 256ull * 1024 * 1024;
  uint64_t p = mm.alloc(n);
  const uint64_t before = resident_bytes();
  const uint8_t v = 0xC3;
  mm.fill(p, &v, 1, n);
  const uint64_t after = resident_bytes();
  const uint64_t grew = after > before ? after - before : 0;
  VCHECK(grew >= n / 2);            // it really did materialize the range
  VCHECK(grew < n + (n / 2));       // but nowhere near twice it
  mm.free(p);
}
