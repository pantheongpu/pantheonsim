// Unit tests for the virtual device memory manager.
#include "vgpu/memory.hpp"

#include <cstdio>
#include <cstring>
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

VTEST_MAIN
