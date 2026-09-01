// Unit tests for the virtual device memory manager.
#include "vgpu/memory.hpp"

#include <cstring>
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
