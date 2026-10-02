// Host codecs for the simulator's nvCOMP: the standard bitstreams nvCOMP's
// chunk formats are, written from their public specifications.
//
//   LZ4      the LZ4 block format (lz4_Block_format.md)
//   Snappy   the raw Snappy format (format_description.txt)
//   Deflate  RFC 1951 raw DEFLATE; gzip is RFC 1952 around it
//   zstd     RFC 8878 Zstandard frames
//
// Each codec has a compressor whose output any conforming decoder reads, and
// a decoder that accepts every conforming stream (not only this compressor's).
// Decoders never read or write out of bounds: a stream that runs past its
// input, refers outside its output, or needs more room than the caller gave
// is reported, never followed.
#ifndef VGPU_NVCOMP_CODECS_HPP
#define VGPU_NVCOMP_CODECS_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vgpu::codec {

using Bytes = std::vector<uint8_t>;

enum class Result {
  Ok,
  Corrupt,      // not a valid stream of this format
  NoRoom,       // valid, but the decoded data does not fit the output
  BadChecksum,  // decoded, and the stream's own checksum disagrees
};

// A decoder writes at most `cap` bytes to `out` and sets *produced. With
// out == nullptr it only measures: *produced is the decoded size.
// ---- LZ4 block ----
size_t lz4_bound(size_t n);
Bytes lz4_compress(const uint8_t* in, size_t n);
Result lz4_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced);

// ---- Snappy (raw) ----
size_t snappy_bound(size_t n);
Bytes snappy_compress(const uint8_t* in, size_t n);
Result snappy_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced);
// The length Snappy's preamble declares.
Result snappy_length(const uint8_t* in, size_t n, size_t* length);

// ---- DEFLATE (raw), and gzip around it ----
// effort 0 is Huffman coding alone, without matches; 1..5 search harder.
size_t deflate_bound(size_t n);
Bytes deflate_compress(const uint8_t* in, size_t n, int effort);
Result inflate(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced, size_t* consumed = nullptr);
size_t gzip_bound(size_t n);
Bytes gzip_compress(const uint8_t* in, size_t n, int effort);
// One or more gzip members, concatenated, each with its CRC and length checked.
Result gunzip(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced);

// ---- Zstandard ----
size_t zstd_bound(size_t n);
Bytes zstd_compress(const uint8_t* in, size_t n);
// One or more frames (skippable frames are skipped), content checksums checked.
Result zstd_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced);

// ---- pieces shared with the GDeflate codec ----
namespace detail {
// An output sink that either writes into the caller's buffer or only counts.
struct Sink {
  uint8_t* out;
  size_t cap;
  size_t n = 0;
  bool room(size_t k) const { return !out || (k <= cap && n <= cap - k); }
  void put(uint8_t b) {
    if (out) out[n] = b;
    ++n;
  }
  void put(const uint8_t* p, size_t k) {
    if (out && k) std::memcpy(out + n, p, k);
    n += k;
  }
  void fill(uint8_t b, size_t k) {
    if (out && k) std::memset(out + n, b, k);
    n += k;
  }
  // A back-reference: `dist` bytes back, `len` bytes, overlapping allowed.
  // The caller has checked 0 < dist <= n.
  void copy(size_t dist, size_t len) {
    if (out) {
      uint8_t* d = out + n;
      const uint8_t* s = d - dist;
      if (dist >= len) std::memcpy(d, s, len);
      else
        for (size_t i = 0; i < len; ++i) d[i] = s[i];
    }
    n += len;
  }
};

struct Token {  // a literal (dist == 0) or a match of lit_or_len bytes, dist back
  uint32_t lit_or_len;
  uint32_t dist;
};
std::vector<Token> lz77(const uint8_t* in, size_t n, int effort, size_t window, size_t max_len);
std::vector<uint8_t> huffman_lengths(std::vector<uint32_t> freq, int limit);
std::vector<uint16_t> canonical_codes(const std::vector<uint8_t>& len);
}  // namespace detail

// ---- checksums ----
uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0);
uint64_t xxh64(const uint8_t* p, size_t n, uint64_t seed = 0);

}  // namespace vgpu::codec

#endif  // VGPU_NVCOMP_CODECS_HPP
