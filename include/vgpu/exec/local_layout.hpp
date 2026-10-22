// Where a lane's local-memory bytes live in a warp's array (the PTX interpreter's; the SASS executor keeps one frame per lane,
// see "Local memory layout" in docs/performance.md for why).
//
// A warp's local memory is one array for all its lanes, laid out the way the hardware lays it out: by 4-byte words, so the
// word `w` of lane `l` is at `(w * lanes + l) * 4`. Byte `o` of lane `l` is then at
//
//     (o / 4) * lanes * 4  +  l * 4  +  o % 4.
//
// A warp whose lanes all access the same offset (the usual case: a spilled register, a per-thread array indexed by the
// loop counter) then touches `lanes * 4` contiguous bytes, two cache lines, where the layout "lane `l`'s frame is
// `local[l * size .. (l + 1) * size)`" touched one line per lane. Nothing but this file knows the layout: the interpreter
// reads and writes through these functions, so what a lane sees at an offset is the same bytes as in the per-lane layout
// (tests/unit/test_locallayout.cpp checks every access shape against a per-lane reference).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vgpu::exec {

// The address of the 4-byte word holding offset `off` (rounded down to the word) of `lane`.
inline uint8_t* local_word(uint8_t* base, uint32_t lanes, uint32_t lane, uint64_t off) {
  return base + (off >> 2) * (uint64_t{lanes} * 4) + uint64_t{lane} * 4;
}
inline const uint8_t* local_word(const uint8_t* base, uint32_t lanes, uint32_t lane, uint64_t off) {
  return base + (off >> 2) * (uint64_t{lanes} * 4) + uint64_t{lane} * 4;
}

// `n` bytes at any offset, byte by byte where they cross a word.
inline void local_read(const uint8_t* base, uint32_t lanes, uint32_t lane, uint64_t off, void* out, uint32_t n) {
  uint8_t* o = static_cast<uint8_t*>(out);
  while (n) {
    const uint32_t in_word = static_cast<uint32_t>(off & 3), take = n < 4 - in_word ? n : 4 - in_word;
    std::memcpy(o, local_word(base, lanes, lane, off) + in_word, take);
    o += take; off += take; n -= take;
  }
}
inline void local_write(uint8_t* base, uint32_t lanes, uint32_t lane, uint64_t off, const void* in, uint32_t n) {
  const uint8_t* s = static_cast<const uint8_t*>(in);
  while (n) {
    const uint32_t in_word = static_cast<uint32_t>(off & 3), take = n < 4 - in_word ? n : 4 - in_word;
    std::memcpy(local_word(base, lanes, lane, off) + in_word, s, take);
    s += take; off += take; n -= take;
  }
}

// Up to 8 bytes at any offset as a little-endian integer (what load_bytes / memcpy into a u64 gives on the per-lane layout).
// The sizes are constants to the compiler in each case (a move, not a call to memcpy with a length); an access that is not
// inside one word, or not 8 bytes at a word boundary, is the byte-wise local_read.
inline uint64_t local_load(const uint8_t* base, uint32_t lanes, uint32_t lane, uint64_t off, uint32_t n) {
  const uint32_t in_word = static_cast<uint32_t>(off & 3);
  const uint8_t* p = local_word(base, lanes, lane, off);
  switch (n) {
    case 1: { uint8_t v; std::memcpy(&v, p + in_word, 1); return v; }
    case 2:
      if (in_word <= 2) { uint16_t v; std::memcpy(&v, p + in_word, 2); return v; }
      break;
    case 4:
      if (in_word == 0) { uint32_t v; std::memcpy(&v, p, 4); return v; }
      break;
    case 8:
      if (in_word == 0) {
        uint32_t lo, hi;
        std::memcpy(&lo, p, 4);
        std::memcpy(&hi, p + uint64_t{lanes} * 4, 4);
        return lo | (uint64_t{hi} << 32);
      }
      break;
    default: break;
  }
  uint64_t v = 0;
  local_read(base, lanes, lane, off, &v, n);
  return v;
}
inline void local_store(uint8_t* base, uint32_t lanes, uint32_t lane, uint64_t off, uint32_t n, uint64_t value) {
  const uint32_t in_word = static_cast<uint32_t>(off & 3);
  uint8_t* p = local_word(base, lanes, lane, off);
  switch (n) {
    case 1: { const uint8_t v = static_cast<uint8_t>(value); std::memcpy(p + in_word, &v, 1); return; }
    case 2:
      if (in_word <= 2) { const uint16_t v = static_cast<uint16_t>(value); std::memcpy(p + in_word, &v, 2); return; }
      break;
    case 4:
      if (in_word == 0) { const uint32_t v = static_cast<uint32_t>(value); std::memcpy(p, &v, 4); return; }
      break;
    case 8:
      if (in_word == 0) {
        const uint32_t lo = static_cast<uint32_t>(value), hi = static_cast<uint32_t>(value >> 32);
        std::memcpy(p, &lo, 4);
        std::memcpy(p + uint64_t{lanes} * 4, &hi, 4);
        return;
      }
      break;
    default: break;
  }
  local_write(base, lanes, lane, off, &value, n);
}

// The size of the array that holds `bytes_per_lane` bytes for each of `lanes` lanes (rounded up to whole words).
inline size_t local_array_bytes(uint32_t lanes, uint64_t bytes_per_lane) {
  return static_cast<size_t>((bytes_per_lane + 3) / 4) * lanes * 4;
}

// Growing the array to hold more bytes per lane is appending zero words: the rows already there stay where they are.

}  // namespace vgpu::exec
