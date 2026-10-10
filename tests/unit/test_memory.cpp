// Unit tests for the virtual device memory manager.
#include "vgpu/memory.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <memory>
#include <thread>
#include <unistd.h>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory_backing.hpp"
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

// AMD's HIP allocates by the 4 KB page: allocations start on one, and a
// kernel's read runs on to the end of the last (hipSPARSELt reads a byte past
// its compressed matrix). Copies and writes past the end are still refused,
// and without a page size a kernel's read past the end is too.
VTEST(page_size_lets_a_kernel_read_the_rest_of_the_page) {
  MemoryManager mm(1 << 20);
  mm.set_page_size(4096);
  const uint64_t p = mm.alloc(4608), q = mm.alloc(4);
  VCHECK_EQ(p % 4096, 0ull);
  VCHECK_EQ(q, p + 8192);
  mm.store_scalar(p + 4604, 4, 0x11223344);
  VCHECK_EQ(mm.load_scalar(p + 4604, 4), 0x11223344ull);
  VCHECK_EQ(mm.load_scalar(p + 4608, 1), 0ull);      // the rest of the page reads
  VCHECK_EQ(mm.load_scalar(p + 8188, 4), 0ull);
  VCHECK(VCAPTURE(Error, mm.store_scalar(p + 4608, 1, 7)).code() == Err::OutOfBounds);
  std::vector<uint8_t> buf(8);
  VCHECK(VCAPTURE(Error, mm.read(p + 4604, buf.data(), buf.size())).code() == Err::OutOfBounds);
  mm.free(q);
  VCHECK(VCAPTURE(Error, mm.load_scalar(q + 8, 4)).code() == Err::UseAfterFree);

  // Without a page size the next allocation starts right after (4608 is a
  // multiple of 256), and a read past the end is not this allocation's.
  MemoryManager strict(1 << 20);
  const uint64_t r = strict.alloc(4608);
  VCHECK_CONTAINS(VCAPTURE(Error, strict.load_scalar(r + 4608, 1)).what(), "device memory read");
  VCHECK_EQ(strict.alloc(4), r + 4608);
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


// --- fill(): memset without a host staging buffer -------------------------
// A memset used to build a host copy the size of the range and then write it
// in, so setting N bytes of device memory cost N bytes of chunks plus N bytes
// of staging, live at the same time. These pin the behaviour that replaced it.


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

// A 2D fill's rows start wherever the pitch puts them, so a row can begin a
// byte or two before a chunk boundary. The pattern's phase belongs to the fill's
// own start, not to the chunk or to alignment -- a 2-byte fill of 0xABCD that
// starts two bytes early still writes CD AB CD AB across the boundary.
VTEST(fill_phase_follows_its_own_start_across_a_chunk_boundary) {
  MemoryManager mm(1 << 20);
  const uint64_t n = vgpu::kChunkSize * 2;
  uint64_t p = mm.alloc(n);
  std::vector<uint8_t> seed(n, 0x11);
  mm.write(p, seed.data(), n);
  const uint16_t half = 0xABCD;
  const uint32_t word = 0x12345678u;
  const uint64_t start2 = vgpu::kChunkSize - 3;   // odd, and just before the boundary
  const uint64_t start4 = vgpu::kChunkSize + 1001;
  mm.fill(p + start2, reinterpret_cast<const uint8_t*>(&half), 2, 10);
  mm.fill(p + start4, reinterpret_cast<const uint8_t*>(&word), 4, 12);
  std::vector<uint8_t> out(n, 0);
  mm.read(p, out.data(), n);
  const uint8_t* h = reinterpret_cast<const uint8_t*>(&half);
  const uint8_t* w = reinterpret_cast<const uint8_t*>(&word);
  for (uint64_t i = 0; i < n; ++i) {
    uint8_t want = 0x11;
    if (i >= start2 && i < start2 + 10) want = h[(i - start2) % 2];
    if (i >= start4 && i < start4 + 12) want = w[(i - start4) % 4];
    VCHECK_EQ(out[i], want);
  }
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
  // of a large device and zeroes it from pulling the whole device into memory.
  MemoryManager mm(1ull << 40);
  const uint64_t n = 256ull * 1024 * 1024;
  uint64_t p = mm.alloc(n);
  VCHECK_EQ(mm.resident_bytes(), 0ull);
  const uint8_t zero = 0;
  mm.fill(p, &zero, 1, n);
  VCHECK_EQ(mm.resident_bytes(), 0ull);  // not one chunk
  // It still reads back as zero.
  std::vector<uint8_t> out(4096, 0xEE);
  mm.read(p + n - 4096, out.data(), out.size());
  for (uint8_t b : out) VCHECK_EQ(b, 0);
  mm.free(p);
}

VTEST(nonzero_fill_costs_about_one_copy) {
  // The staging buffer made a fill of N bytes cost about 2N of host memory.
  // Backing is exactly one chunk per touched chunk now, so this is an equality
  // rather than a range: the earlier version compared the process's resident
  // size, which a sanitizer's shadow memory and the allocator both perturb.
  // A one-byte fill costs nothing at all (whole chunks become uniform), so the
  // copy is measured with a four-byte pattern, which needs real bytes.
  MemoryManager mm(1ull << 40);
  const uint64_t n = 256ull * 1024 * 1024;
  uint64_t p = mm.alloc(n);
  const uint8_t pattern[4] = {0xC3, 0x3C, 0x5A, 0xA5};
  mm.fill(p, pattern, 4, n);
  VCHECK_EQ(mm.resident_bytes(), n);  // one copy, not two
  VCHECK_EQ(mm.load_scalar(p + n - 4, 4), uint64_t{0xA55A3CC3});
  // A second fill over the same range materializes nothing further.
  mm.fill(p, pattern, 4, n);
  VCHECK_EQ(mm.resident_bytes(), n);
  // One byte over it gives every chunk back.
  const uint8_t v = 0xC3;
  mm.fill(p, &v, 1, n);
  VCHECK_EQ(mm.resident_bytes(), 0ull);
  mm.free(p);
  VCHECK_EQ(mm.resident_bytes(), 0ull);
}

VTEST(a_length_that_wraps_the_end_address_is_still_out_of_bounds) {
  // Computed as end = addr + len, a length near UINT64_MAX wraps to a small
  // number and compares *below* the allocation's end -- so the check passed
  // exactly the access it exists to stop. It is a remaining-length compare now.
  MemoryManager mem(1 << 20);
  uint64_t p = mem.alloc(64);
  std::vector<uint8_t> buf(64);
  auto err = VCAPTURE(Error, mem.read(p + 32, buf.data(), ~uint64_t{0} - 16));
  VCHECK(err.code() == Err::OutOfBounds);
}

VTEST(null_host_pointers_are_diagnosed_not_dereferenced) {
  MemoryManager mem(1 << 20);
  uint64_t p = mem.alloc(64);
  auto w = VCAPTURE(Error, mem.write(p, nullptr, 8));
  VCHECK(w.code() == Err::InvalidPointer);
  VCHECK_CONTAINS(w.what(), "NULL host pointer");
  auto r = VCAPTURE(Error, mem.read(p, nullptr, 8));
  VCHECK(r.code() == Err::InvalidPointer);
  VCHECK_CONTAINS(r.what(), "NULL host pointer");
}

// ---- managed memory ----
//
// One buffer the host dereferences directly and a kernel also addresses. Every
// other device pointer here is a virtual address with no host meaning, which
// is why cudaMallocManaged used to refuse; a managed buffer is real host
// memory put on the device's map.

VTEST(a_mapped_host_buffer_is_readable_and_writable_from_the_device_side) {
  MemoryManager mem(1 << 20);
  std::vector<uint32_t> host(64, 0);
  const uint64_t addr = reinterpret_cast<uint64_t>(host.data());
  mem.map_host(addr, host.data(), host.size() * sizeof(uint32_t));

  // The device side writes; the host sees it without a copy.
  mem.store_scalar(addr + 4 * 4, 4, 0xABCDu);
  VCHECK_EQ(host[4], 0xABCDu);
  // The host writes; the device side sees it without a copy.
  host[9] = 0x1234u;
  VCHECK_EQ(mem.load_scalar(addr + 9 * 4, 4), uint64_t{0x1234u});

  mem.unmap_host(addr);
}

VTEST(host_buffers_mapped_end_to_end_read_write_and_fill_as_one_range) {
  // HIP's virtual memory maps separate host buffers side by side at device
  // addresses; a copy or a fill that runs across the seam reaches both.
  MemoryManager mem(1 << 20);
  std::vector<uint8_t> a(4096, 0), b(4096, 0);
  const uint64_t at = 0x7e00'0000'0000ull;
  mem.map_host(at, a.data(), a.size());
  mem.map_host(at + a.size(), b.data(), b.size());
  std::vector<uint8_t> in(64);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i + 1);
  mem.write(at + 4096 - 32, in.data(), in.size());
  VCHECK_EQ(a[4095], 32u);
  VCHECK_EQ(b[0], 33u);
  std::vector<uint8_t> out(64, 0);
  mem.read(at + 4096 - 32, out.data(), out.size());
  VCHECK(out == in);
  const uint8_t pattern[2] = {0xAA, 0xBB};
  mem.fill(at + 4096 - 3, pattern, 2, 6);   // the phase runs on across the seam
  VCHECK_EQ(a[4093], 0xAAu);
  VCHECK_EQ(a[4095], 0xAAu);
  VCHECK_EQ(b[0], 0xBBu);
  VCHECK_EQ(b[2], 0xBBu);
  VCHECK_EQ(b[3], 36u);   // past the fill: what the write left
  mem.unmap_host(at);
  mem.unmap_host(at + a.size());
}

VTEST(a_mapped_buffer_is_owned_by_the_device_even_outside_its_window) {
  // owns() is what decides whether a pointer is treated as device memory, and
  // a managed buffer lives at its real host address -- outside every device
  // VA window -- while still being device-addressable.
  MemoryManager mem(1 << 20);
  std::vector<uint32_t> host(16, 0);
  const uint64_t addr = reinterpret_cast<uint64_t>(host.data());
  VCHECK(!mem.owns(addr));
  mem.map_host(addr, host.data(), host.size() * sizeof(uint32_t));
  VCHECK(mem.owns(addr));
  VCHECK(mem.is_host_mapped(addr));
  mem.unmap_host(addr);
  // And once unmapped it is not device memory again, so a kernel cannot reach
  // a pointer the process has given back to the allocator.
  VCHECK(!mem.owns(addr));
  VCHECK(!mem.is_host_mapped(addr));
}

VTEST(ordinary_device_memory_is_unaffected_by_a_mapping_existing) {
  // The mapped-range check sits on the read and write path, so it has to be
  // invisible to everything that is not managed.
  MemoryManager mem(1 << 20);
  std::vector<uint32_t> host(16, 7);
  const uint64_t mapped = reinterpret_cast<uint64_t>(host.data());
  mem.map_host(mapped, host.data(), host.size() * sizeof(uint32_t));

  const uint64_t dev = mem.alloc(256);
  mem.store_scalar(dev, 4, 0x5555u);
  VCHECK_EQ(mem.load_scalar(dev, 4), uint64_t{0x5555u});
  VCHECK_EQ(host[0], 7u);  // untouched by the device-memory write
  mem.free(dev);
  mem.unmap_host(mapped);
}

// Host buffers either side of the device windows (a heap one low, an mmap
// one high) make the mapped range span every device address; device accesses
// must still reach device memory (and no longer take the host-map lock). A
// host buffer mapped inside a window, however unlikely, must still be found.
VTEST(host_maps_around_and_inside_the_device_windows) {
  MemoryManager mem(1 << 20);
  std::vector<uint32_t> low(16, 1), high(16, 2), inside(16, 3);
  mem.map_host(0x10000, low.data(), low.size() * sizeof(uint32_t));
  const uint64_t hi_addr = 0x7f00'0000'0000ull;
  mem.map_host(hi_addr, high.data(), high.size() * sizeof(uint32_t));
  const uint64_t dev = mem.alloc(256);
  mem.store_scalar(dev + 8, 4, 0x77u);
  VCHECK_EQ(mem.load_scalar(dev + 8, 4), uint64_t{0x77u});
  uint32_t v = 0;
  mem.read(dev + 8, &v, 4);
  VCHECK_EQ(v, 0x77u);
  VCHECK_EQ(mem.load_scalar(hi_addr + 4, 4), uint64_t{2u});
  // The last device window, which this device does not own.
  const uint64_t in_window = vgpu::kDeviceVaEnd - vgpu::kDeviceVaStride + 0x1000;
  mem.map_host(in_window, inside.data(), inside.size() * sizeof(uint32_t));
  VCHECK_EQ(mem.load_scalar(in_window + 8, 4), uint64_t{3u});
  mem.store_scalar(in_window + 12, 4, 0x99u);
  VCHECK_EQ(inside[3], 0x99u);
  VCHECK_EQ(mem.load_scalar(dev + 8, 4), uint64_t{0x77u});
  mem.free(dev);
  mem.unmap_host(in_window);
  mem.unmap_host(hi_addr);
  mem.unmap_host(0x10000);
}

// load_run / store_run (TMA's box rows) must do exactly what element-by-
// element load_scalar / store_scalar do: across a chunk boundary, over
// untouched and uniform chunks, from any element-aligned start; and decline,
// doing nothing, whatever they cannot answer that way.
VTEST(load_run_and_store_run_match_scalar_accesses) {
  MemoryManager mem(1 << 22);
  const uint64_t n = 3 * vgpu::kChunkSize;
  const uint64_t p = mem.alloc(n);
  const uint8_t b = 0x5A;
  mem.fill(p + vgpu::kChunkSize, &b, 1, vgpu::kChunkSize);        // chunk 1 uniform, chunk 0 untouched
  for (uint64_t i = 0; i < 64; i += 2) mem.store_scalar(p + 2 * vgpu::kChunkSize + i, 2, 0x1000 + i);
  for (uint32_t unit : {1u, 2u, 4u, 8u}) {
    // From a unit-aligned start 24 bytes before chunk 1 to 40 into chunk 2.
    const uint64_t at = p + vgpu::kChunkSize - 24, len = vgpu::kChunkSize + 64;
    std::vector<uint8_t> got(len);
    VCHECK(mem.load_run(at, got.data(), len, unit));
    for (uint64_t i = 0; i < len; i += unit) {
      uint64_t want = mem.load_scalar(at + i, unit), have = 0;
      std::memcpy(&have, got.data() + i, unit);
      if (have != want) VCHECK_EQ(have, want);
    }
  }
  // Storing a uniform chunk's own byte leaves it alone.
  std::vector<uint8_t> same(64, 0x5A);
  VCHECK(mem.store_run(p + vgpu::kChunkSize, same.data(), same.size(), 8));
  VCHECK_EQ(mem.load_scalar(p + vgpu::kChunkSize + 8, 8), uint64_t{0x5A5A5A5A5A5A5A5Aull});
  // Stores: an element-aligned run of 2-byte values across the uniform
  // chunk's end into the next.
  std::vector<uint8_t> src(200);
  for (size_t i = 0; i < src.size(); ++i) src[i] = uint8_t(i * 7 + 1);
  const uint64_t to = p + 2 * vgpu::kChunkSize - 100;
  VCHECK(mem.store_run(to, src.data(), src.size(), 2));
  for (size_t i = 0; i < src.size(); i += 2)
    VCHECK_EQ(mem.load_scalar(to + i, 2), uint64_t(src[i] | src[i + 1] << 8));
  VCHECK_EQ(mem.load_scalar(to - 2, 2), uint64_t{0x5A5A});   // the uniform bytes before it kept
  // Declined: misaligned, past the end, and outside any allocation.
  uint8_t tmp[16];
  VCHECK(!mem.load_run(p + 1, tmp, 8, 2));
  VCHECK(!mem.load_run(p + n - 8, tmp, 16, 8));
  VCHECK(!mem.store_run(p + n - 8, tmp, 16, 8));
  VCHECK(!mem.load_run(0x1000, tmp, 8, 8));
  mem.free(p);
}

// Blocks run on several host threads and may share global memory: CUB's
// decoupled look-back publishes a block's prefix for later blocks to spin on.
// A kernel's aligned scalar access is atomic on a GPU, so it must be here --
// never half of one store and half of another. Under ThreadSanitizer this is
// also the check that the access is not a data race.
VTEST(concurrent_scalar_accesses_are_never_torn) {
  MemoryManager mm(1 << 20);
  const uint64_t p = mm.alloc(64);
  mm.store_scalar(p, 8, 0);
  constexpr uint64_t kA = 0, kB = ~0ull;
  constexpr int kIters = 20000;
  std::atomic<bool> torn{false};
  std::thread writer([&] {
    for (int i = 0; i < kIters; ++i) mm.store_scalar(p, 8, (i & 1) ? kB : kA);
  });
  std::thread reader([&] {
    for (int i = 0; i < kIters; ++i) {
      const uint64_t v = mm.load_scalar(p, 8);
      if (v != kA && v != kB) torn = true;
    }
  });
  writer.join();
  reader.join();
  VCHECK(!torn.load());
  mm.free(p);
}

// The look-back pattern itself: one block writes its aggregate, fences, and
// sets a flag; another spins on the flag, fences, and must then see the
// aggregate. The fences are the kernel's (membar.gl / fence.acq_rel).
VTEST(a_fenced_flag_publishes_the_data_written_before_it) {
  MemoryManager mm(1 << 20);
  const uint64_t data = mm.alloc(64), flag = mm.alloc(64);
  std::thread producer([&] {
    mm.store_scalar(data, 4, 0x1234abcd);
    std::atomic_thread_fence(std::memory_order_release);
    mm.store_scalar(flag, 4, 1);
  });
  uint64_t seen = 0;
  std::thread consumer([&] {
    while (mm.load_scalar(flag, 4) == 0) std::this_thread::yield();
    std::atomic_thread_fence(std::memory_order_acquire);
    seen = mm.load_scalar(data, 4);
  });
  producer.join();
  consumer.join();
  VCHECK_EQ(seen, uint64_t{0x1234abcd});
  mm.free(data);
  mm.free(flag);
}

// Two blocks storing into the same untouched chunk at once both materialize
// it. Only one copy survives, and neither store may land in the discarded one.
VTEST(first_stores_racing_into_a_fresh_chunk_both_survive) {
  for (int round = 0; round < 200; ++round) {
    MemoryManager mm(1 << 20);
    const uint64_t p = mm.alloc(4096);
    std::thread a([&] { mm.store_scalar(p, 8, 0xAAAAAAAAAAAAAAAAull); });
    std::thread b([&] { mm.store_scalar(p + 2048, 8, 0xBBBBBBBBBBBBBBBBull); });
    a.join();
    b.join();
    VCHECK_EQ(mm.load_scalar(p, 8), 0xAAAAAAAAAAAAAAAAull);
    VCHECK_EQ(mm.load_scalar(p + 2048, 8), 0xBBBBBBBBBBBBBBBBull);
  }
}

// A load from memory nothing has written reads zero without creating backing
// storage; only a store materializes.
VTEST(a_scalar_load_of_untouched_memory_is_zero_and_allocates_nothing) {
  MemoryManager mm(1 << 20);
  const uint64_t p = mm.alloc(1 << 16);
  VCHECK_EQ(mm.load_scalar(p + 8, 8), uint64_t{0});
  VCHECK_EQ(mm.load_scalar(p + 3, 1), uint64_t{0});
  VCHECK_EQ(mm.resident_bytes(), uint64_t{0});
  mm.store_scalar(p + 6, 2, 0xbeef);
  VCHECK_EQ(mm.load_scalar(p + 6, 2), uint64_t{0xbeef});
  VCHECK_EQ(mm.load_scalar(p + 4, 4), uint64_t{0xbeef0000});
  VCHECK_EQ(mm.resident_bytes(), vgpu::kChunkSize);
  mm.free(p);
}

// A load of memory nothing has written, racing the first store into it. The
// load used to find the chunk unmaterialized and fall back to a plain copy --
// and the store could materialize and write the chunk in between, so the copy
// read bytes mid-write: undefined, and reported by ThreadSanitizer on main.
// Untouched memory now answers zero without copying anything. Many rounds,
// because the window is a few instructions wide.
VTEST(a_load_racing_the_first_store_into_untouched_memory_is_never_a_copy) {
  bool wrong = false;
  for (int round = 0; round < 300 && !wrong; ++round) {
    MemoryManager mm(1 << 20);
    const uint64_t p = mm.alloc(64);
    std::atomic<bool> go{false};
    std::thread writer([&] {
      while (!go.load()) std::this_thread::yield();
      mm.store_scalar(p, 4, 0xdecafbad);
    });
    std::thread reader([&] {
      go = true;
      for (;;) {
        const uint64_t v = mm.load_scalar(p, 4);
        if (v == 0xdecafbad) break;
        if (v != 0) { wrong = true; break; }
      }
    });
    writer.join();
    reader.join();
    mm.free(p);
  }
  VCHECK(!wrong);
}

// A scalar whose storage is not aligned for its width still round-trips: it
// is read and written a byte at a time rather than copied.
VTEST(scalar_store_and_load_agree_at_every_size) {
  MemoryManager mm(1 << 20);
  const uint64_t p = mm.alloc(64);
  mm.store_scalar(p + 8, 8, 0x0123456789abcdefull);
  mm.store_scalar(p + 20, 4, 0xa1b2c3d4);
  mm.store_scalar(p + 26, 2, 0xbeef);
  mm.store_scalar(p + 31, 1, 0x7f);
  VCHECK_EQ(mm.load_scalar(p + 8, 8), 0x0123456789abcdefull);
  VCHECK_EQ(mm.load_scalar(p + 20, 4), uint64_t{0xa1b2c3d4});
  VCHECK_EQ(mm.load_scalar(p + 26, 2), uint64_t{0xbeef});
  VCHECK_EQ(mm.load_scalar(p + 31, 1), uint64_t{0x7f});
  VCHECK_EQ(mm.load_scalar(p + 12, 4), uint64_t{0x01234567});   // the high half of the u64
  mm.free(p);
}

// A card larger than the machine's RAM: chunks past the RAM limit live in a
// file, read back exactly, and give their disk space back when freed.
VTEST(device_memory_past_the_ram_limit_spills_to_disk) {
  using vgpu::kChunkSize;
  char dir[] = "/tmp/vgpu-spill-XXXXXX";
  VCHECK(::mkdtemp(dir) != nullptr);
  vgpu::backing::configure({4 * kChunkSize, dir});
  const uint64_t ram0 = vgpu::backing::ram_bytes();
  {
    MemoryManager mm(1ull << 30);
    const uint64_t n = 64 * kChunkSize;  // 4 MiB: 4 chunks in RAM, 60 on disk
    const uint64_t p = mm.alloc(n);
    std::vector<uint8_t> in(n), out(n, 0);
    for (uint64_t i = 0; i < n; ++i) in[i] = static_cast<uint8_t>(i * 131 + 7);
    mm.write(p, in.data(), n);
    VCHECK_EQ(vgpu::backing::ram_bytes() - ram0, 4 * kChunkSize);
    VCHECK_EQ(vgpu::backing::spill_bytes(), 60 * kChunkSize);
    mm.read(p, out.data(), n);
    VCHECK(in == out);
    // Scalar access goes straight to the chunk, file or not.
    mm.store_scalar(p + 40 * kChunkSize + 16, 8, 0x1122334455667788ull);
    VCHECK_EQ(mm.load_scalar(p + 40 * kChunkSize + 16, 8), 0x1122334455667788ull);
    mm.free(p);
    VCHECK_EQ(vgpu::backing::spill_bytes(), 0ull);
    VCHECK_EQ(vgpu::backing::ram_bytes(), ram0);

    // Chunks handed out again read as zero, as untouched memory does.
    const uint64_t q = mm.alloc(n);
    const uint8_t one = 1;
    for (uint64_t c = 0; c < 64; ++c) mm.write(q + c * kChunkSize, &one, 1);
    std::vector<uint8_t> again(n, 0xEE);
    mm.read(q, again.data(), n);
    bool rest_zero = true;
    for (uint64_t i = 0; i < n; ++i)
      if (again[i] != (i % kChunkSize == 0 ? 1 : 0)) rest_zero = false;
    VCHECK(rest_zero);
    mm.free(q);
  }
  vgpu::backing::configure({});
  ::rmdir(dir);
}

// Nowhere to spill is a CUDA out-of-memory error, not a crash.
VTEST(device_memory_with_nowhere_to_spill_is_out_of_memory) {
  vgpu::backing::configure({0, "/nonexistent/vgpu-spill"});
  {
    MemoryManager mm(1 << 20);
    const uint64_t p = mm.alloc(64);
    bool oom = false;
    try {
      mm.store_scalar(p, 8, 42);
    } catch (const Error& e) {
      oom = e.code() == Err::OutOfMemory;
    }
    VCHECK(oom);
    mm.free(p);
  }
  vgpu::backing::configure({});
}

// A card filled with one byte costs no memory: a stress test's 12 GB fill is a
// table of tags. Reads see the byte, storing it again changes nothing, and only
// a chunk written with something else becomes real.
VTEST(a_uniform_fill_costs_nothing_until_other_data_is_written) {
  using vgpu::kChunkSize;
  MemoryManager mm(64ull << 30);
  const uint64_t n = 1ull << 30;  // 1 GiB
  const uint64_t p = mm.alloc(n);
  const uint8_t ff = 0xFF;
  mm.fill(p, &ff, 1, n);
  VCHECK_EQ(mm.resident_bytes(), 0ull);
  VCHECK_EQ(mm.load_scalar(p + 12345 * 4, 4), uint64_t{0xFFFFFFFF});
  VCHECK_EQ(mm.load_scalar(p + n - 8, 8), ~uint64_t{0});
  std::vector<uint8_t> out(3 * kChunkSize, 0);
  mm.read(p + kChunkSize / 2, out.data(), out.size());
  bool all_ff = true;
  for (uint8_t b : out) all_ff = all_ff && b == 0xFF;
  VCHECK(all_ff);

  // The fill pattern, stored or written again: still nothing.
  for (uint64_t i = 0; i < 1000; ++i) mm.store_scalar(p + i * 4096, 4, 0xFFFFFFFF);
  const std::vector<uint8_t> again(4096, 0xFF);
  mm.write(p + 7 * kChunkSize + 100, again.data(), again.size());
  VCHECK_EQ(mm.resident_bytes(), 0ull);

  // Something else: one real chunk, the rest of it still the fill byte.
  mm.store_scalar(p + 5 * kChunkSize + 16, 4, 0x12345678);
  VCHECK_EQ(mm.resident_bytes(), kChunkSize);
  VCHECK_EQ(mm.load_scalar(p + 5 * kChunkSize + 16, 4), uint64_t{0x12345678});
  VCHECK_EQ(mm.load_scalar(p + 5 * kChunkSize + 20, 4), uint64_t{0xFFFFFFFF});
  VCHECK_EQ(mm.load_scalar(p + 6 * kChunkSize, 4), uint64_t{0xFFFFFFFF});

  // Filled again, the real chunk goes back; filled with zero, it is untouched memory.
  mm.fill(p, &ff, 1, n);
  VCHECK_EQ(mm.resident_bytes(), 0ull);
  const uint8_t zero = 0;
  mm.write(p, again.data(), 16);
  mm.store_scalar(p + 64, 8, 42);
  mm.fill(p, &zero, 1, n);
  VCHECK_EQ(mm.resident_bytes(), 0ull);
  VCHECK_EQ(mm.load_scalar(p + 64, 8), 0ull);
  mm.free(p);
}

// A fill that does not cover a chunk exactly still lands byte for byte.
VTEST(a_partial_fill_over_a_uniform_chunk_keeps_both_values) {
  using vgpu::kChunkSize;
  MemoryManager mm(1ull << 30);
  const uint64_t p = mm.alloc(4 * kChunkSize);
  const uint8_t a = 0xAA, b = 0x55;
  mm.fill(p, &a, 1, 4 * kChunkSize);
  mm.fill(p + kChunkSize + 10, &b, 1, 20);
  VCHECK_EQ(mm.load_scalar(p + kChunkSize + 8, 1), uint64_t{0xAA});
  VCHECK_EQ(mm.load_scalar(p + kChunkSize + 10, 1), uint64_t{0x55});
  VCHECK_EQ(mm.load_scalar(p + kChunkSize + 29, 1), uint64_t{0x55});
  VCHECK_EQ(mm.load_scalar(p + kChunkSize + 30, 1), uint64_t{0xAA});
  VCHECK_EQ(mm.resident_bytes(), kChunkSize);
  mm.free(p);
}

// Blocks writing different values into the same uniform chunk at once all land.
VTEST(stores_racing_into_a_uniform_chunk_all_land) {
  using vgpu::kChunkSize;
  MemoryManager mm(1ull << 30);
  const uint64_t p = mm.alloc(kChunkSize);
  const uint8_t ff = 0xFF;
  mm.fill(p, &ff, 1, kChunkSize);
  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < 8; ++t)
    threads.emplace_back([&, t] {
      for (uint64_t i = 0; i < 512; ++i) mm.store_scalar(p + (t * 1024 + i) * 8, 8, t * 100000 + i);
    });
  for (auto& th : threads) th.join();
  bool ok = true;
  for (uint32_t t = 0; t < 8; ++t)
    for (uint64_t i = 0; i < 512; ++i) ok = ok && mm.load_scalar(p + (t * 1024 + i) * 8, 8) == t * 100000 + i;
  VCHECK(ok);
  VCHECK_EQ(mm.resident_bytes(), kChunkSize);
  mm.free(p);
}

// With the RAM limit at nothing and a spill directory, a uniform fill still
// touches neither.
VTEST(a_uniform_fill_never_spills) {
  using vgpu::kChunkSize;
  char dir[] = "/tmp/vgpu-uniform-XXXXXX";
  VCHECK(::mkdtemp(dir) != nullptr);
  vgpu::backing::configure({0, dir});
  {
    MemoryManager mm(64ull << 30);
    const uint64_t n = 256 * kChunkSize;
    const uint64_t p = mm.alloc(n);
    const uint8_t v = 0x5A;
    mm.fill(p, &v, 1, n);
    VCHECK_EQ(vgpu::backing::spill_bytes(), 0ull);
    VCHECK_EQ(mm.load_scalar(p + n - 1, 1), uint64_t{0x5A});
    mm.free(p);
  }
  vgpu::backing::configure({});
  ::rmdir(dir);
}

// A forked child that frees its copy of spilled device memory, or exits and
// runs its destructors, must not zero the parent's: the spill file is shared.
VTEST(a_forked_child_freeing_spilled_memory_leaves_the_parents_intact) {
  using vgpu::kChunkSize;
  char dir[] = "/tmp/vgpu-fork-XXXXXX";
  VCHECK(::mkdtemp(dir) != nullptr);
  vgpu::backing::configure({0, dir});
  {
    MemoryManager mm(1ull << 30);
    const uint64_t n = 16 * kChunkSize;
    const uint64_t p = mm.alloc(n);
    std::vector<uint8_t> in(n), out(n);
    for (uint64_t i = 0; i < n; ++i) in[i] = static_cast<uint8_t>(i * 7 + 1);
    mm.write(p, in.data(), n);
    const pid_t child = ::fork();
    if (child == 0) {
      mm.free(p);
      // And a fresh spill of its own, which must not reuse the parent's chunks.
      const uint64_t q = mm.alloc(4 * kChunkSize);
      std::vector<uint8_t> junk(4 * kChunkSize, 0xEE);
      mm.write(q, junk.data(), junk.size());
      ::_exit(0);
    }
    int status = 0;
    VCHECK(::waitpid(child, &status, 0) == child);
    VCHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    mm.read(p, out.data(), n);
    VCHECK(in == out);
    // The parent's own spill still works and still frees.
    const uint64_t r = mm.alloc(4 * kChunkSize);
    mm.write(r, in.data(), 4 * kChunkSize);
    mm.free(r);
    mm.free(p);
    VCHECK_EQ(vgpu::backing::spill_bytes(), 0ull);
  }
  vgpu::backing::configure({});
  ::rmdir(dir);
}

// Device windows must not cover host memory. They began at 0x7fff'0000'0000,
// where the stack lives, and a stack buffer was taken for a device pointer.
VTEST(stack_and_heap_addresses_are_never_device_addresses) {
  int on_stack = 0;
  const auto heap = std::make_unique<int>(0);
  VCHECK(!vgpu::is_device_va(reinterpret_cast<uint64_t>(&on_stack)));
  VCHECK(!vgpu::is_device_va(reinterpret_cast<uint64_t>(heap.get())));
  VCHECK(!vgpu::is_device_va(0x7fff'4d6e'2190ull));   // the stack address from the flaky copy
  VCHECK(!vgpu::is_device_va(0));
  MemoryManager last(1 << 20, static_cast<uint32_t>(vgpu::kDeviceVaWindows - 1));
  const uint64_t p = last.alloc(64);
  VCHECK(vgpu::is_device_va(p));
  VCHECK(vgpu::is_device_va(vgpu::kDeviceVaEnd - 1));
  VCHECK(!vgpu::is_device_va(vgpu::kDeviceVaEnd));
  last.free(p);
}

// Kernels read the allocation table on every access, from many threads, and
// with asynchronous streams the host may allocate, free and map while they
// do. Readers here hammer allocations that stay put while a writer thread
// allocates and frees others: every read must see its own bytes, and the
// ThreadSanitizer job must find nothing.
VTEST(allocating_and_freeing_while_other_threads_read_is_safe) {
  MemoryManager mm(256ull << 20);
  constexpr int kReaders = 6, kFixed = 16;
  std::vector<uint64_t> fixed(kFixed);
  for (int i = 0; i < kFixed; ++i) {
    fixed[i] = mm.alloc(4096);
    for (uint64_t w = 0; w < 4096; w += 8) mm.store_scalar(fixed[i] + w, 8, (uint64_t{static_cast<unsigned>(i)} << 32) | w);
  }
  std::atomic<bool> stop{false};
  std::atomic<int> wrong{0};
  std::vector<std::thread> readers;
  for (int t = 0; t < kReaders; ++t)
    readers.emplace_back([&, t] {
      uint64_t n = static_cast<uint64_t>(t);
      while (!stop.load(std::memory_order_relaxed)) {
        const int i = static_cast<int>(n % kFixed);
        const uint64_t w = (n * 8) % 4096;
        uint64_t base = 0, size = 0;
        if (mm.load_scalar(fixed[i] + w, 8) != ((uint64_t{static_cast<unsigned>(i)} << 32) | w) ||
            !mm.find_allocation(fixed[i] + w, &base, &size) || base != fixed[i] || size != 4096)
          wrong.fetch_add(1);
        ++n;
      }
    });
  // The writer: allocations of every size come and go, and a reservation is
  // made, mapped, unmapped and given back, all while the readers run.
  for (int round = 0; round < 400; ++round) {
    std::vector<uint64_t> tmp;
    for (int j = 1; j <= 8; ++j) tmp.push_back(mm.alloc(static_cast<uint64_t>(j) * 1000));
    for (uint64_t p : tmp) mm.store_scalar(p, 4, 7);
    for (uint64_t p : tmp) mm.free(p);
    const uint64_t va = mm.reserve(MemoryManager::kVmmGranularity, 0), h = mm.create_handle(MemoryManager::kVmmGranularity);
    mm.map(va, MemoryManager::kVmmGranularity, 0, h);
    mm.unmap(va, MemoryManager::kVmmGranularity);
    mm.release_handle(h);
    mm.address_free(va, MemoryManager::kVmmGranularity);
  }
  stop = true;
  for (auto& r : readers) r.join();
  VCHECK_EQ(wrong.load(), 0);
}

// The lock under the table: many readers at once, a writer that waits for
// them and holds them off, and a writer that takes it again and reads under
// it from its own thread (the table's mutators call one another).
VTEST(the_big_reader_lock_keeps_writers_and_readers_apart) {
  vgpu::BigReaderLock lock;
  int value = 0;          // written only under the writer's lock
  std::atomic<int> torn{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t)
    readers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        vgpu::SharedGuard g(&lock);
        const int a = value, b = value;   // a writer mid-update would show here
        if (a != b || a % 2) torn.fetch_add(1);
      }
    });
  for (int i = 0; i < 2000; ++i) {
    vgpu::ExclusiveGuard g(&lock);
    ++value;              // odd while the writer holds it
    {
      vgpu::ExclusiveGuard again(&lock);   // re-entrant for its owner
      vgpu::SharedGuard read(&lock);       // and it may read under it
      ++value;
    }
  }
  stop = true;
  for (auto& r : readers) r.join();
  VCHECK_EQ(torn.load(), 0);
  VCHECK_EQ(value, 4000);
}

VTEST_MAIN

// ---- virtual memory management -------------------------------------------------
//
// Address space, physical handles and the mapping between them. Each test is a
// state a real device distinguishes: reserved with nothing mapped, mapped with
// no access, read-only, unmapped again.

VTEST(reserved_address_space_has_nothing_behind_it) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(4 * g, 0);
  VCHECK(va >= vgpu::kDeviceVaBase && va % g == 0);
  VCHECK_EQ(mm.reservations(), size_t{1});
  VCHECK_EQ(mm.used(), uint64_t{0});   // address space costs no memory
  uint8_t byte = 0;
  auto err = VCAPTURE(Error, mm.read(va, &byte, 1));
  VCHECK(err.code() == Err::InvalidPointer);
  VCHECK_CONTAINS(err.what(), "reserved address space with nothing mapped");
}

VTEST(a_handle_is_memory_without_an_address) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t h = mm.create_handle(2 * g);
  VCHECK(h != 0);
  VCHECK_EQ(mm.handle_size(h), 2 * g);
  VCHECK_EQ(mm.used(), 2 * g);          // memory is charged at create, not at map
  VCHECK_EQ(mm.handles(), size_t{1});
  mm.release_handle(h);
  VCHECK_EQ(mm.used(), uint64_t{0});
  VCHECK_EQ(mm.handles(), size_t{0});
  auto err = VCAPTURE(Error, mm.handle_size(h));
  VCHECK(err.code() == Err::InvalidValue);
}

VTEST(mapped_memory_is_unusable_until_access_is_granted) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(2 * g, 0), h = mm.create_handle(g);
  mm.map(va, g, 0, h);
  uint8_t byte = 0;
  auto err = VCAPTURE(Error, mm.read(va, &byte, 1));
  VCHECK_CONTAINS(err.what(), "no device has been given access");
  mm.set_access(va, g, /*readable=*/true, /*writable=*/true);
  std::vector<uint8_t> in{1, 2, 3, 4}, out(4, 0);
  mm.write(va, in.data(), in.size());
  mm.read(va, out.data(), out.size());
  VCHECK(out == in);
  bool r = false, w = false;
  VCHECK(mm.access_at(va, &r, &w) && r && w);
}

VTEST(a_read_only_mapping_refuses_a_write) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(g, 0), h = mm.create_handle(g);
  mm.map(va, g, 0, h);
  mm.set_access(va, g, /*readable=*/true, /*writable=*/false);
  uint8_t byte = 7;
  auto err = VCAPTURE(Error, mm.write(va, &byte, 1));
  VCHECK_CONTAINS(err.what(), "read-only mapping");
  mm.read(va, &byte, 1);               // reading is allowed
  VCHECK_EQ(static_cast<int>(byte), 0);
  bool r = false, w = true;
  VCHECK(mm.access_at(va, &r, &w) && r && !w);
}

VTEST(the_same_memory_shows_through_every_address_it_is_mapped_at) {
  MemoryManager mm(8 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t a = mm.reserve(g, 0), b = mm.reserve(g, 0), h = mm.create_handle(g);
  mm.map(a, g, 0, h);
  mm.map(b, g, 0, h);
  mm.set_access(a, g, true, true);
  mm.set_access(b, g, true, true);
  const uint8_t in[4] = {9, 8, 7, 6};
  mm.write(a, in, sizeof in);
  uint8_t out[4] = {0, 0, 0, 0};
  mm.read(b, out, sizeof out);          // one allocation, two addresses
  VCHECK(std::memcmp(in, out, sizeof in) == 0);
  VCHECK_EQ(mm.used(), g);              // charged once, not twice
}

VTEST(unmapping_takes_the_memory_away_from_the_address) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(g, 0), h = mm.create_handle(g);
  mm.map(va, g, 0, h);
  mm.set_access(va, g, true, true);
  uint8_t byte = 5;
  mm.write(va, &byte, 1);
  mm.unmap(va, g);
  auto err = VCAPTURE(Error, mm.read(va, &byte, 1));
  VCHECK_CONTAINS(err.what(), "reserved address space with nothing mapped");
  VCHECK_EQ(mm.mappings(), size_t{0});
  // The handle still holds the memory: nothing has released it.
  VCHECK_EQ(mm.used(), g);
  mm.release_handle(h);
  VCHECK_EQ(mm.used(), uint64_t{0});
}

// CUDA lets a handle be released while it is still mapped; the mapping keeps
// working and the memory goes when the last mapping does.
VTEST(a_released_handle_stays_alive_while_it_is_mapped) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(g, 0), h = mm.create_handle(g);
  mm.map(va, g, 0, h);
  mm.set_access(va, g, true, true);
  mm.release_handle(h);
  VCHECK_EQ(mm.used(), g);
  uint8_t byte = 3;
  mm.write(va, &byte, 1);
  byte = 0;
  mm.read(va, &byte, 1);
  VCHECK_EQ(static_cast<int>(byte), 3);
  mm.unmap(va, g);
  VCHECK_EQ(mm.used(), uint64_t{0});
}

VTEST(vmm_refuses_what_a_device_would) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  VCHECK(VCAPTURE(Error, mm.reserve(g + 1, 0)).code() == Err::InvalidValue);        // size
  VCHECK(VCAPTURE(Error, mm.reserve(g, 3 * g / 2)).code() == Err::InvalidValue);    // alignment: not a power of two
  VCHECK(VCAPTURE(Error, mm.create_handle(g - 1)).code() == Err::InvalidValue);
  VCHECK(VCAPTURE(Error, mm.create_handle(64 << 20)).code() == Err::OutOfMemory);   // past capacity
  uint64_t va = mm.reserve(2 * g, 0), h = mm.create_handle(g);
  VCHECK_CONTAINS(VCAPTURE(Error, mm.map(va, g, g, h)).what(), "offset of 0");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.map(va, g, 0, h + 99)).what(), "does not exist");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.map(va, 2 * g, 0, h)).what(), "which holds");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.map(va + 8, g, 0, h)).what(), "multiples of the");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.map(va + 4 * g, g, 0, h)).what(), "not inside one reservation");
  mm.map(va, g, 0, h);
  VCHECK_CONTAINS(VCAPTURE(Error, mm.map(va, g, 0, h)).what(), "already mapped");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.unmap(va, 2 * g)).what(), "no mapping of exactly");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.set_access(va, 2 * g, true, true)).what(),
                  "not the start of a mapping");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.address_free(va, 2 * g)).what(), "still mapped");
  mm.set_access(va, g, true, true);
  // An access that starts inside the mapping and runs past it is an overrun,
  // not an unmapped address.
  std::vector<uint8_t> big(static_cast<size_t>(g) + 16, 0);
  VCHECK(VCAPTURE(Error, mm.read(va, big.data(), big.size())).code() == Err::OutOfBounds);
  mm.unmap(va, g);
  mm.release_handle(h);
  mm.address_free(va, 2 * g);
  VCHECK_EQ(mm.reservations(), size_t{0});
}

VTEST(the_granularity_is_two_mib_and_a_smaller_alignment_is_raised_to_it) {
  MemoryManager mm(16 << 20);
  VCHECK_EQ(MemoryManager::kVmmGranularity, uint64_t{2} << 20);   // what an RTX 3060 reports, minimum and recommended
  const uint64_t g = MemoryManager::kVmmGranularity;
  // An alignment below the granule is accepted and every reservation is aligned to a granule at least, as on the
  // card; a power of two above it is honoured; one that is not a power of two is refused.
  for (uint64_t alignment : {uint64_t{0}, uint64_t{4096}, uint64_t{64} << 10, g, 2 * g, uint64_t{1} << 30}) {
    const uint64_t va = mm.reserve(g, alignment);
    VCHECK_EQ(va % g, uint64_t{0});
    if (alignment > g) VCHECK_EQ(va % alignment, uint64_t{0});
    mm.address_free(va, g);
  }
  VCHECK(VCAPTURE(Error, mm.reserve(g, 5)).code() == Err::InvalidValue);
  VCHECK(VCAPTURE(Error, mm.reserve(g, 3 * g)).code() == Err::InvalidValue);
  VCHECK(VCAPTURE(Error, mm.reserve(64 << 10, 0)).code() == Err::InvalidValue);     // smaller than a granule
  VCHECK(VCAPTURE(Error, mm.create_handle(64 << 10)).code() == Err::InvalidValue);
  VCHECK(VCAPTURE(Error, mm.create_handle(g + 1)).code() == Err::InvalidValue);
}

VTEST(a_range_is_mapped_when_any_part_of_it_is) {
  MemoryManager mm(16 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  const uint64_t va = mm.reserve(4 * g, 0), h = mm.create_handle(g);
  VCHECK(!mm.range_mapped(va, 4 * g));
  mm.map(va + g, g, 0, h);
  VCHECK(mm.range_mapped(va, 4 * g));
  VCHECK(mm.range_mapped(va + g, g));
  VCHECK(mm.range_mapped(va + g + g / 2, 1));     // inside the mapping
  VCHECK(!mm.range_mapped(va, g));                // before it
  VCHECK(!mm.range_mapped(va + 2 * g, 2 * g));    // after it
  mm.unmap(va + g, g);
  VCHECK(!mm.range_mapped(va, 4 * g));
  mm.release_handle(h);
  mm.address_free(va, 4 * g);
}

VTEST(vmm_unmaps_a_run_of_whole_mappings_in_one_call) {
  // ggml's VMM pool maps a chunk at a time and unmaps the whole pool at once.
  MemoryManager mm(8 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(4 * g, 0);
  uint64_t h[3];
  for (int i = 0; i < 3; ++i) {
    h[i] = mm.create_handle(g);
    mm.map(va + i * g, g, 0, h[i]);
  }
  // Not a run of whole mappings: it ends inside one, or reaches past the last.
  VCHECK_CONTAINS(VCAPTURE(Error, mm.unmap(va, g / 2)).what(), "no mapping of exactly");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.unmap(va, 4 * g)).what(), "no mapping of exactly");
  VCHECK_CONTAINS(VCAPTURE(Error, mm.unmap(va + g / 2, g)).what(), "no mapping of exactly");
  mm.unmap(va, 3 * g);
  // All three are gone: each can be mapped again, and the handles are free to go.
  mm.map(va, g, 0, h[0]);
  mm.unmap(va, g);
  for (int i = 0; i < 3; ++i) mm.release_handle(h[i]);
  mm.address_free(va, 4 * g);
  VCHECK_EQ(mm.reservations(), size_t{0});
}

VTEST(a_device_reset_takes_mappings_reservations_and_handles) {
  MemoryManager mm(4 << 20);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(g, 0), h = mm.create_handle(g);
  mm.map(va, g, 0, h);
  mm.set_access(va, g, true, true);
  mm.free_all();
  VCHECK_EQ(mm.mappings(), size_t{0});
  VCHECK_EQ(mm.reservations(), size_t{0});
  VCHECK_EQ(mm.handles(), size_t{0});
  VCHECK_EQ(mm.used(), uint64_t{0});
}

// Only chunks that were written cost host memory, mapped or not: the sparse
// backing is what lets a handle be as large as the card.
VTEST(a_mapped_handle_is_as_sparse_as_any_other_allocation) {
  MemoryManager mm(1ull << 32);
  const uint64_t g = MemoryManager::kVmmGranularity;
  uint64_t va = mm.reserve(1ull << 30, 0), h = mm.create_handle(1ull << 30);
  mm.map(va, 1ull << 30, 0, h);
  mm.set_access(va, 1ull << 30, true, true);
  const uint8_t one = 1;
  mm.write(va + (1ull << 29), &one, 1);   // one byte, in the middle
  VCHECK(mm.resident_bytes() <= 2 * g);
  VCHECK_EQ(mm.used(), 1ull << 30);       // the card's own accounting is the full size
}

// ---- memory another process can map ---------------------------------------------
//
// The mechanics, without a second process: an exported allocation keeps working
// at the same address, its bytes move into a file, and a mapping of that file
// reads them back. e2e_ipc is the test that runs two real processes.

VTEST(an_exported_allocation_keeps_its_address_and_its_contents) {
  MemoryManager mm(4 << 20);
  const std::string path = std::string("/tmp/vgpu-share-test-") + std::to_string(getpid());
  std::filesystem::remove(path);
  uint64_t p = mm.alloc(4096);
  std::vector<uint8_t> in(4096);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint8_t>(i * 3 + 1);
  mm.write(p, in.data(), in.size());

  VCHECK_EQ(mm.share(p, path), uint64_t{4096});
  VCHECK(mm.is_shared(p));
  VCHECK(std::filesystem::exists(path));
  // The same address, the same bytes, and writes still land.
  std::vector<uint8_t> out(4096, 0);
  mm.read(p, out.data(), out.size());
  VCHECK(out == in);
  const uint8_t marker = 0x5a;
  mm.write(p + 8, &marker, 1);
  uint8_t got = 0;
  mm.read(p + 8, &got, 1);
  VCHECK_EQ(static_cast<int>(got), 0x5a);
  // Sharing twice is the same share.
  VCHECK_EQ(mm.share(p, path + "-again"), uint64_t{4096});
  VCHECK(!std::filesystem::exists(path + "-again"));

  // What another process would see: the file holds the bytes.
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  VCHECK_EQ(file.size(), size_t{4096});
  VCHECK_EQ(static_cast<int>(file[8]), 0x5a);
  f.close();

  // Freeing it takes the file with it: the exporting process owns it.
  mm.free(p);
  VCHECK(!std::filesystem::exists(path));
  VCHECK(!mm.is_shared(p));
  VCHECK_EQ(mm.used(), uint64_t{0});
}

VTEST(a_mapping_of_shared_memory_sees_the_same_bytes) {
  MemoryManager owner(4 << 20), other(4 << 20, 1);   // as two processes would have
  const std::string path = std::string("/tmp/vgpu-adopt-test-") + std::to_string(getpid());
  std::filesystem::remove(path);
  uint64_t p = owner.alloc(4096);
  const uint8_t seven = 7;
  owner.write(p + 16, &seven, 1);
  owner.share(p, path);

  const uint64_t mapped = other.adopt(path, 4096);
  VCHECK(mapped != p);            // a different address, as IPC gives
  uint8_t got = 0;
  other.read(mapped + 16, &got, 1);
  VCHECK_EQ(static_cast<int>(got), 7);
  // A write through one mapping shows in the other: this is shared memory.
  const uint8_t nine = 9;
  other.write(mapped + 32, &nine, 1);
  owner.read(p + 32, &got, 1);
  VCHECK_EQ(static_cast<int>(got), 9);

  other.abandon(mapped);
  // Abandoning what was not adopted, or adopting what is not there, is refused.
  VCHECK(VCAPTURE(Error, other.abandon(mapped)).code() == Err::InvalidPointer);
  VCHECK(VCAPTURE(Error, owner.abandon(p)).code() == Err::InvalidPointer);
  VCHECK(VCAPTURE(Error, other.adopt(path + "-missing", 4096)).code() == Err::InvalidValue);
  VCHECK(VCAPTURE(Error, owner.share(p + 64, path + "-interior")).code() == Err::InvalidPointer);
  owner.free(p);
  VCHECK(!std::filesystem::exists(path));
}

// Sharing a large allocation of which little was written keeps the file
// sparse: only the chunks that hold something are copied into it. Copying
// the zeros too made every page of a 1 GiB NVSHMEM heap real, and filled the
// small tmpfs the IPC files live on.
VTEST(sharing_keeps_untouched_memory_sparse) {
  MemoryManager mm(512ull << 20);
  const std::string path = std::string("/tmp/vgpu-sparse-share-test-") + std::to_string(getpid());
  std::filesystem::remove(path);
  const uint64_t size = 256ull << 20;
  const uint64_t p = mm.alloc(size);
  const uint8_t v = 0x42;
  mm.write(p + (100ull << 20), &v, 1);   // one byte, deep inside
  mm.share(p, path);
  struct stat sb {};
  VCHECK(::stat(path.c_str(), &sb) == 0);
  VCHECK_EQ(uint64_t(sb.st_size), size);
  VCHECK(uint64_t(sb.st_blocks) * 512 < (8ull << 20));   // a chunk or so, not 256 MiB
  uint8_t got = 0;
  mm.read(p + (100ull << 20), &got, 1);
  VCHECK_EQ(int(got), 0x42);
  mm.read(p + (200ull << 20), &got, 1);
  VCHECK_EQ(int(got), 0);
  // Once every importer has it mapped, the exporter can drop the name and
  // keep the memory; an address it did not share is refused.
  VCHECK(mm.unlink_shared(p));
  VCHECK(!std::filesystem::exists(path));
  VCHECK(!mm.unlink_shared(p + 4096));
  mm.read(p + (100ull << 20), &got, 1);
  VCHECK_EQ(int(got), 0x42);
  mm.free(p);
}

// ---- the device heap (malloc and free in a kernel) -----------------------

// The heap's limit is a budget of bytes asked for: up to it a block, past it
// null rather than an error, and a request so large the sum would wrap is
// past it too.
VTEST(the_device_heap_returns_null_past_its_limit) {
  MemoryManager mm(64 << 20);
  const uint64_t limit = 8 << 20;
  const uint64_t a = mm.heap_alloc(6 << 20, limit);
  VCHECK(a != 0);
  VCHECK_EQ(mm.heap_used(), uint64_t{6 << 20});
  VCHECK_EQ(mm.heap_alloc(3 << 20, limit), uint64_t{0});
  VCHECK_EQ(mm.heap_alloc(0, limit), uint64_t{0});
  VCHECK_EQ(mm.heap_alloc(~uint64_t{0}, limit), uint64_t{0});
  VCHECK_EQ(mm.heap_alloc(~uint64_t{0} - (1 << 20), limit), uint64_t{0});
  const uint64_t b = mm.heap_alloc(2 << 20, limit);   // exactly the rest
  VCHECK(b != 0);
  VCHECK_EQ(mm.heap_alloc(1, limit), uint64_t{0});
  VCHECK_EQ(mm.heap_used(), limit);
  // Blocks are ordinary allocations, of the size asked for.
  uint64_t base = 0, size = 0;
  VCHECK(mm.find_allocation(b + 100, &base, &size));
  VCHECK_EQ(base, b);
  VCHECK_EQ(size, uint64_t{2 << 20});
  VCHECK_EQ(mm.used(), limit);
}

// free() gives the bytes back to the budget; a pointer the heap did not hand
// out -- cudaMalloc's, an interior one, one already freed -- is refused with
// nothing changed, for the engine to report.
VTEST(the_device_heap_refunds_a_free_and_refuses_what_it_did_not_allocate) {
  MemoryManager mm(64 << 20);
  const uint64_t limit = 8 << 20;
  const uint64_t a = mm.heap_alloc(6 << 20, limit);
  const uint64_t host = mm.alloc(4096);
  VCHECK(!mm.heap_free(host));
  uint64_t base = 0, size = 0;
  VCHECK(mm.find_allocation(host, &base, &size));   // still live
  VCHECK(!mm.heap_free(a + 256));
  VCHECK_EQ(mm.heap_used(), uint64_t{6 << 20});
  VCHECK(mm.heap_free(a));
  VCHECK_EQ(mm.heap_used(), uint64_t{0});
  VCHECK(!mm.heap_free(a));   // twice
  // The freed block is quarantined like any other: a use after free is named.
  uint8_t byte = 0;
  VCHECK(VCAPTURE(Error, mm.read(a, &byte, 1)).code() == Err::UseAfterFree);
  VCHECK(mm.heap_alloc(8 << 20, limit) != 0);   // the whole budget is back
  mm.free(host);
}

// A device reset frees the heap's blocks with everything else and gives the
// budget back, so blocks a program leaked before it do not count after.
VTEST(a_device_reset_empties_the_device_heap) {
  MemoryManager mm(64 << 20);
  const uint64_t limit = 8 << 20;
  uint64_t leaked[3];
  for (uint64_t& p : leaked) {
    p = mm.heap_alloc(2 << 20, limit);
    VCHECK(p != 0);
  }
  mm.free_all();
  VCHECK_EQ(mm.heap_used(), uint64_t{0});
  VCHECK_EQ(mm.live_allocations(), size_t{0});
  for (uint64_t p : leaked) VCHECK(!mm.heap_free(p));   // not the heap's any more
  VCHECK(mm.heap_alloc(limit - 256, limit) != 0);
}

// When the device's memory runs out before the heap's limit, malloc still
// returns null rather than failing.
VTEST(the_device_heap_returns_null_when_device_memory_is_full) {
  MemoryManager mm(1 << 20);
  VCHECK_EQ(mm.heap_alloc(2 << 20, 8 << 20), uint64_t{0});
  VCHECK_EQ(mm.heap_used(), uint64_t{0});
  VCHECK(mm.heap_alloc(1 << 20, 8 << 20) != 0);
}

// Blocks run on several host threads: however their mallocs interleave, the
// heap hands out exactly its limit and no more.
VTEST(the_device_heap_holds_its_limit_across_threads) {
  MemoryManager mm(256 << 20);
  const uint64_t limit = 1 << 20, each = 4096;
  std::atomic<uint64_t> got{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t)
    threads.emplace_back([&] {
      for (int i = 0; i < 64; ++i)
        if (mm.heap_alloc(each, limit)) got += each;
    });
  for (std::thread& t : threads) t.join();
  VCHECK_EQ(got.load(), limit);
  VCHECK_EQ(mm.heap_used(), limit);
}
