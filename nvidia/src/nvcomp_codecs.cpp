// Host codecs for the simulator's nvCOMP (see nvcomp_codecs.hpp).
//
// Written from the formats' public specifications:
//   LZ4 block format      github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
//   Snappy                github.com/google/snappy/blob/main/format_description.txt
//   DEFLATE, gzip         RFC 1951, RFC 1952
//   Zstandard             RFC 8878
//   XXH64                 github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
#include "nvcomp_codecs.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <queue>
#include <utility>

namespace vgpu::codec {
namespace {

using detail::Sink;

inline uint32_t rd16(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8; }
inline uint32_t rd24(const uint8_t* p) { return rd16(p) | uint32_t(p[2]) << 16; }
inline uint32_t rd32(const uint8_t* p) { return rd16(p) | rd16(p + 2) << 16; }
inline uint64_t rd64(const uint8_t* p) { return uint64_t(rd32(p)) | uint64_t(rd32(p + 4)) << 32; }

inline int highbit(uint64_t v) {  // index of the highest set bit; v != 0
  return 63 - __builtin_clzll(v);
}

// Literal and match accounting shared by the LZ-style compressors.
struct HashTable {
  std::vector<int64_t> slot;
  int bits;
  explicit HashTable(int b) : slot(size_t(1) << b, -1), bits(b) {}
  size_t index(uint32_t v) const { return (v * 2654435761u) >> (32 - bits); }
};

}  // namespace

// ============================================================================
// Checksums
// ============================================================================

uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

uint64_t xxh64(const uint8_t* p, size_t n, uint64_t seed) {
  constexpr uint64_t P1 = 11400714785074694791ull, P2 = 14029467366897019727ull, P3 = 1609587929392839161ull,
                     P4 = 9650029242287828579ull, P5 = 2870177450012600261ull;
  auto rotl = [](uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
  auto round = [&](uint64_t acc, uint64_t lane) { return rotl(acc + lane * P2, 31) * P1; };
  auto merge = [&](uint64_t acc, uint64_t v) { return (acc ^ round(0, v)) * P1 + P4; };
  const uint8_t* end = p + n;
  uint64_t h;
  if (n >= 32) {
    uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
    while (end - p >= 32) {
      v1 = round(v1, rd64(p));
      v2 = round(v2, rd64(p + 8));
      v3 = round(v3, rd64(p + 16));
      v4 = round(v4, rd64(p + 24));
      p += 32;
    }
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1);
    h = merge(h, v2);
    h = merge(h, v3);
    h = merge(h, v4);
  } else {
    h = seed + P5;
  }
  h += n;
  while (end - p >= 8) {
    h ^= round(0, rd64(p));
    h = rotl(h, 27) * P1 + P4;
    p += 8;
  }
  if (end - p >= 4) {
    h ^= uint64_t(rd32(p)) * P1;
    h = rotl(h, 23) * P2 + P3;
    p += 4;
  }
  while (p < end) {
    h ^= uint64_t(*p++) * P5;
    h = rotl(h, 11) * P1;
  }
  h ^= h >> 33;
  h *= P2;
  h ^= h >> 29;
  h *= P3;
  h ^= h >> 32;
  return h;
}

// ============================================================================
// LZ4 block format
// ============================================================================

// NVIDIA's bound (nvcompBatchedLZ4CompressGetMaxOutputChunkSize, measured on
// an RTX 3060 from 0 to 16 MiB); a block of literals alone needs
// n + n / 255 + 2 at most.
size_t lz4_bound(size_t n) { return (n + n / 255 + 9) & ~size_t(7); }

namespace {
void lz4_length(Bytes& o, size_t v) {  // the 255-run continuation of a 15 nibble
  while (v >= 255) {
    o.push_back(255);
    v -= 255;
  }
  o.push_back(uint8_t(v));
}
void lz4_sequence(Bytes& o, const uint8_t* lit, size_t nlit, size_t offset, size_t mlen) {
  const size_t m = mlen ? mlen - 4 : 0;
  o.push_back(uint8_t((std::min<size_t>(nlit, 15) << 4) | std::min<size_t>(m, 15)));
  if (nlit >= 15) lz4_length(o, nlit - 15);
  o.insert(o.end(), lit, lit + nlit);
  if (!mlen) return;  // the last sequence: literals only
  o.push_back(uint8_t(offset));
  o.push_back(uint8_t(offset >> 8));
  if (m >= 15) lz4_length(o, m - 15);
}
}  // namespace

// Greedy matching on a 4-byte hash. The block format's end rules: the last
// five bytes are literals, and no match starts in the last twelve.
Bytes lz4_compress(const uint8_t* in, size_t n) {
  Bytes o;
  o.reserve(lz4_bound(n));
  constexpr size_t kLastLiterals = 5, kMfLimit = 12, kMaxOffset = 65535;
  size_t anchor = 0;
  if (n > kMfLimit) {
    HashTable ht(16);
    const size_t match_limit = n - kLastLiterals;
    size_t ip = 0;
    while (ip + kMfLimit < n) {
      const uint32_t seq = rd32(in + ip);
      int64_t& slot = ht.slot[ht.index(seq)];
      const int64_t ref = slot;
      slot = (int64_t)ip;
      if (ref < 0 || ip - (size_t)ref > kMaxOffset || rd32(in + ref) != seq) {
        ++ip;
        continue;
      }
      size_t len = 4;
      while (ip + len < match_limit && in[ref + len] == in[ip + len]) ++len;
      // Extend backwards over literals that also match.
      size_t back = 0;
      while (ip - back > anchor && (size_t)ref > back && in[ip - back - 1] == in[ref - back - 1]) ++back;
      lz4_sequence(o, in + anchor, ip - back - anchor, ip - (size_t)ref, len + back);
      ip += len;
      anchor = ip;
      if (ip >= 2 && ip + 4 <= n) ht.slot[ht.index(rd32(in + ip - 2))] = (int64_t)(ip - 2);
    }
  }
  lz4_sequence(o, in + anchor, n - anchor, 0, 0);
  return o;
}

Result lz4_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced) {
  Sink s{out, cap};
  size_t ip = 0;
  *produced = 0;
  auto ext = [&](size_t& v) -> bool {  // a 255-run continuation
    for (;;) {
      if (ip >= n) return false;
      const uint8_t b = in[ip++];
      v += b;
      if (b != 255) return true;
    }
  };
  for (;;) {
    if (ip >= n) return Result::Corrupt;
    const uint8_t token = in[ip++];
    size_t lit = token >> 4;
    if (lit == 15 && !ext(lit)) return Result::Corrupt;
    if (lit > n - ip) return Result::Corrupt;
    if (!s.room(lit)) return Result::NoRoom;
    s.put(in + ip, lit);
    ip += lit;
    if (ip == n) break;  // the last sequence has no match
    if (n - ip < 2) return Result::Corrupt;
    const size_t off = rd16(in + ip);
    ip += 2;
    if (off == 0 || off > s.n) return Result::Corrupt;
    size_t ml = token & 15;
    if (ml == 15 && !ext(ml)) return Result::Corrupt;
    ml += 4;
    if (!s.room(ml)) return Result::NoRoom;
    s.copy(off, ml);
  }
  *produced = s.n;
  return Result::Ok;
}

// ============================================================================
// Snappy (raw format)
// ============================================================================

size_t snappy_bound(size_t n) { return 32 + n + n / 6; }

namespace {
void snappy_literal(Bytes& o, const uint8_t* p, size_t len) {
  while (len) {
    // One element carries any length; split anyway at 64 KiB to keep the
    // length field small.
    const size_t k = std::min<size_t>(len, 65536);
    const size_t m = k - 1;
    if (m < 60) {
      o.push_back(uint8_t(m << 2));
    } else if (m < 256) {
      o.push_back(60 << 2);
      o.push_back(uint8_t(m));
    } else {
      o.push_back(61 << 2);
      o.push_back(uint8_t(m));
      o.push_back(uint8_t(m >> 8));
    }
    o.insert(o.end(), p, p + k);
    p += k;
    len -= k;
  }
}
void snappy_copy(Bytes& o, size_t offset, size_t len) {
  while (len) {
    // A copy with a 2-byte offset carries 1 to 64 bytes; keep every piece but
    // the last at least 4 so that short tails can use the 1-byte-offset form.
    size_t k = std::min<size_t>(len, 64);
    if (len > 64 && len - 64 < 4) k = 60;
    if (k >= 4 && k <= 11 && offset < 2048) {
      o.push_back(uint8_t(1 | ((k - 4) << 2) | ((offset >> 8) << 5)));
      o.push_back(uint8_t(offset));
    } else {
      o.push_back(uint8_t(2 | ((k - 1) << 2)));
      o.push_back(uint8_t(offset));
      o.push_back(uint8_t(offset >> 8));
    }
    len -= k;
  }
}
}  // namespace

Bytes snappy_compress(const uint8_t* in, size_t n) {
  Bytes o;
  o.reserve(snappy_bound(n));
  for (uint64_t v = n;;) {  // the preamble: the length as a varint
    if (v < 128) {
      o.push_back(uint8_t(v));
      break;
    }
    o.push_back(uint8_t(v | 128));
    v >>= 7;
  }
  constexpr size_t kMaxOffset = 65535;
  size_t anchor = 0, ip = 0;
  if (n >= 8) {
    HashTable ht(15);
    while (ip + 4 <= n) {
      const uint32_t seq = rd32(in + ip);
      int64_t& slot = ht.slot[ht.index(seq)];
      const int64_t ref = slot;
      slot = (int64_t)ip;
      if (ref < 0 || ip - (size_t)ref > kMaxOffset || rd32(in + ref) != seq) {
        ++ip;
        continue;
      }
      size_t len = 4;
      while (ip + len < n && in[ref + len] == in[ip + len]) ++len;
      if (ip > anchor) snappy_literal(o, in + anchor, ip - anchor);
      snappy_copy(o, ip - (size_t)ref, len);
      ip += len;
      anchor = ip;
    }
  }
  if (n > anchor) snappy_literal(o, in + anchor, n - anchor);
  return o;
}

Result snappy_length(const uint8_t* in, size_t n, size_t* length) {
  uint64_t v = 0;
  for (size_t i = 0; i < 5; ++i) {
    if (i >= n) return Result::Corrupt;
    v |= uint64_t(in[i] & 127) << (7 * i);
    if (!(in[i] & 128)) {
      if (v > 0xFFFFFFFFull) return Result::Corrupt;
      *length = (size_t)v;
      return Result::Ok;
    }
  }
  return Result::Corrupt;
}

Result snappy_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced) {
  *produced = 0;
  size_t want = 0, ip = 0;
  if (snappy_length(in, n, &want) != Result::Ok) return Result::Corrupt;
  while (in[ip] & 128) ++ip;
  ++ip;
  if (out && want > cap) return Result::NoRoom;
  Sink s{out, cap};
  while (ip < n) {
    const uint8_t tag = in[ip++];
    size_t len = 0, off = 0;
    switch (tag & 3) {
      case 0: {  // literal
        len = tag >> 2;
        if (len >= 60) {
          const size_t bytes = len - 59;
          if (n - ip < bytes) return Result::Corrupt;
          len = 0;
          for (size_t i = 0; i < bytes; ++i) len |= size_t(in[ip + i]) << (8 * i);
          ip += bytes;
        }
        ++len;
        if (len > n - ip || len > want - s.n) return Result::Corrupt;
        s.put(in + ip, len);
        ip += len;
        continue;
      }
      case 1:
        if (n - ip < 1) return Result::Corrupt;
        len = 4 + ((tag >> 2) & 7);
        off = (size_t(tag >> 5) << 8) | in[ip];
        ip += 1;
        break;
      case 2:
        if (n - ip < 2) return Result::Corrupt;
        len = 1 + (tag >> 2);
        off = rd16(in + ip);
        ip += 2;
        break;
      default:
        if (n - ip < 4) return Result::Corrupt;
        len = 1 + (tag >> 2);
        off = rd32(in + ip);
        ip += 4;
        break;
    }
    if (off == 0 || off > s.n || len > want - s.n) return Result::Corrupt;
    s.copy(off, len);
  }
  if (s.n != want) return Result::Corrupt;
  *produced = s.n;
  return Result::Ok;
}

// ============================================================================
// DEFLATE (RFC 1951)
// ============================================================================

namespace {

constexpr uint16_t kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                   31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                   2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr uint16_t kDistBase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                                    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                    6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
constexpr uint8_t kClOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

// ---- decoding ----

struct BitIn {  // LSB-first, as DEFLATE packs bits
  const uint8_t* p;
  size_t n;
  size_t pos = 0;  // in bits
  bool over = false;
  uint32_t bits(int k) {
    uint32_t v = 0;
    for (int i = 0; i < k; ++i) {
      const size_t byte = pos >> 3;
      if (byte >= n) {
        over = true;
        return 0;
      }
      v |= uint32_t((p[byte] >> (pos & 7)) & 1) << i;
      ++pos;
    }
    return v;
  }
  void align() { pos = (pos + 7) & ~size_t(7); }
};

// A canonical Huffman decoder, by code length (RFC 1951 3.2.2).
struct Huff {
  uint16_t count[16] = {};
  uint16_t symbol[320] = {};
  // Builds from lengths; returns <0 for an over-subscribed set, >0 for an
  // incomplete one, 0 for a complete code.
  int build(const uint8_t* len, int n) {
    std::memset(count, 0, sizeof count);
    for (int i = 0; i < n; ++i) count[len[i]]++;
    if (count[0] == n) return 0;  // no codes
    int left = 1;
    for (int l = 1; l < 16; ++l) {
      left <<= 1;
      left -= count[l];
      if (left < 0) return -1;
    }
    uint16_t offs[16];
    offs[1] = 0;
    for (int l = 1; l < 15; ++l) offs[l + 1] = offs[l] + count[l];
    for (int i = 0; i < n; ++i)
      if (len[i]) symbol[offs[len[i]]++] = (uint16_t)i;
    return left;
  }
  int decode(BitIn& b) const {
    int code = 0, first = 0, index = 0;
    for (int l = 1; l < 16; ++l) {
      code |= (int)b.bits(1);
      if (b.over) return -1;
      const int c = count[l];
      if (code - c < first) return symbol[index + (code - first)];
      index += c;
      first += c;
      first <<= 1;
      code <<= 1;
    }
    return -1;
  }
};

Result inflate_block(BitIn& b, Sink& s, const Huff& lit, const Huff& dist) {
  for (;;) {
    const int sym = lit.decode(b);
    if (sym < 0) return Result::Corrupt;
    if (sym < 256) {
      if (!s.room(1)) return Result::NoRoom;
      s.put(uint8_t(sym));
      continue;
    }
    if (sym == 256) return Result::Ok;
    const int li = sym - 257;
    if (li >= 29) return Result::Corrupt;
    const size_t len = kLenBase[li] + b.bits(kLenExtra[li]);
    const int ds = dist.decode(b);
    if (ds < 0 || ds >= 30) return Result::Corrupt;
    const size_t d = kDistBase[ds] + b.bits(kDistExtra[ds]);
    if (b.over || d > s.n) return Result::Corrupt;
    if (!s.room(len)) return Result::NoRoom;
    s.copy(d, len);
  }
}

}  // namespace

Result inflate(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced, size_t* consumed) {
  BitIn b{in, n};
  Sink s{out, cap};
  *produced = 0;
  static const std::pair<Huff, Huff> fixed = [] {
    uint8_t l[288];
    for (int i = 0; i < 144; ++i) l[i] = 8;
    for (int i = 144; i < 256; ++i) l[i] = 9;
    for (int i = 256; i < 280; ++i) l[i] = 7;
    for (int i = 280; i < 288; ++i) l[i] = 8;
    std::pair<Huff, Huff> h;
    h.first.build(l, 288);
    uint8_t d[30];
    std::fill(d, d + 30, 5);
    h.second.build(d, 30);
    return h;
  }();
  int last = 0;
  do {
    last = (int)b.bits(1);
    const uint32_t type = b.bits(2);
    if (b.over) return Result::Corrupt;
    Result r = Result::Ok;
    if (type == 0) {  // stored
      b.align();
      const size_t at = b.pos >> 3;
      if (n < 4 || at > n - 4) return Result::Corrupt;
      const uint32_t len = rd16(in + at), nlen = rd16(in + at + 2);
      if ((len ^ 0xFFFF) != nlen || len > n - at - 4) return Result::Corrupt;
      if (!s.room(len)) return Result::NoRoom;
      s.put(in + at + 4, len);
      b.pos += size_t(4 + len) * 8;
    } else if (type == 1) {
      r = inflate_block(b, s, fixed.first, fixed.second);
    } else if (type == 2) {
      const int nlen = (int)b.bits(5) + 257, ndist = (int)b.bits(5) + 1, ncode = (int)b.bits(4) + 4;
      if (b.over || nlen > 286 || ndist > 30) return Result::Corrupt;
      uint8_t cl[19] = {};
      for (int i = 0; i < ncode; ++i) cl[kClOrder[i]] = (uint8_t)b.bits(3);
      Huff clh;
      if (b.over || clh.build(cl, 19) != 0) return Result::Corrupt;
      uint8_t lens[320] = {};
      for (int i = 0; i < nlen + ndist;) {
        const int sym = clh.decode(b);
        if (sym < 0) return Result::Corrupt;
        if (sym < 16) {
          lens[i++] = (uint8_t)sym;
          continue;
        }
        uint8_t v = 0;
        int rep;
        if (sym == 16) {
          if (i == 0) return Result::Corrupt;
          v = lens[i - 1];
          rep = 3 + (int)b.bits(2);
        } else if (sym == 17) {
          rep = 3 + (int)b.bits(3);
        } else {
          rep = 11 + (int)b.bits(7);
        }
        if (b.over || i + rep > nlen + ndist) return Result::Corrupt;
        while (rep--) lens[i++] = v;
      }
      if (lens[256] == 0) return Result::Corrupt;  // no end-of-block code
      Huff lh, dh;
      const int lr = lh.build(lens, nlen);
      // An incomplete literal/length code is allowed only as a single code.
      if (lr < 0 || (lr > 0 && nlen - lh.count[0] != 1)) return Result::Corrupt;
      const int dr = dh.build(lens + nlen, ndist);
      if (dr < 0 || (dr > 0 && ndist - dh.count[0] != 1)) return Result::Corrupt;
      r = inflate_block(b, s, lh, dh);
    } else {
      return Result::Corrupt;
    }
    if (r != Result::Ok) return r;
  } while (!last);
  if (consumed) *consumed = (b.pos + 7) >> 3;
  *produced = s.n;
  return Result::Ok;
}

// ---- encoding ----

namespace {

struct BitOut {
  Bytes& o;
  uint64_t acc = 0;
  int nacc = 0;
  explicit BitOut(Bytes& out) : o(out) {}
  void put(uint32_t v, int k) {
    acc |= uint64_t(v) << nacc;
    nacc += k;
    while (nacc >= 8) {
      o.push_back(uint8_t(acc));
      acc >>= 8;
      nacc -= 8;
    }
  }
  void flush() {
    if (nacc > 0) o.push_back(uint8_t(acc));
    acc = 0;
    nacc = 0;
  }
};

}  // namespace

namespace detail {

// Code lengths for `freq`, no longer than `limit`: Huffman's construction,
// and if a code comes out too long, the same with the frequencies halved
// until it fits (each halving flattens the tree; equal weights fit in
// ceil(log2(n)) bits, within every limit used here).
std::vector<uint8_t> huffman_lengths(std::vector<uint32_t> freq, int limit) {
  const int n = (int)freq.size();
  std::vector<uint8_t> len(n, 0);
  for (;;) {
    std::vector<int> used;
    for (int i = 0; i < n; ++i)
      if (freq[i]) used.push_back(i);
    std::fill(len.begin(), len.end(), 0);
    if (used.empty()) return len;
    if (used.size() == 1) {
      len[used[0]] = 1;
      return len;
    }
    struct Node {
      uint64_t w;
      int id;
    };
    auto cmp = [](const Node& a, const Node& b) { return a.w > b.w || (a.w == b.w && a.id > b.id); };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> q(cmp);
    std::vector<int> parent(2 * n, -1);
    int next = n;
    for (int i : used) q.push({freq[i], i});
    while (q.size() > 1) {
      const Node a = q.top();
      q.pop();
      const Node b = q.top();
      q.pop();
      parent[a.id] = parent[b.id] = next;
      q.push({a.w + b.w, next++});
    }
    int longest = 0;
    for (int i : used) {
      int d = 0;
      for (int x = i; parent[x] >= 0; x = parent[x]) ++d;
      len[i] = (uint8_t)d;
      longest = std::max(longest, d);
    }
    if (longest <= limit) return len;
    for (int i : used) freq[i] = std::max<uint32_t>(1, freq[i] >> 1);
  }
}

// Canonical codes from lengths, bit-reversed for LSB-first output.
std::vector<uint16_t> canonical_codes(const std::vector<uint8_t>& len) {
  uint16_t bl_count[16] = {}, next[16] = {};
  for (uint8_t l : len) bl_count[l]++;
  bl_count[0] = 0;
  uint16_t code = 0;
  for (int b = 1; b < 16; ++b) {
    code = uint16_t((code + bl_count[b - 1]) << 1);
    next[b] = code;
  }
  std::vector<uint16_t> out(len.size(), 0);
  for (size_t i = 0; i < len.size(); ++i) {
    const int l = len[i];
    if (!l) continue;
    uint16_t c = next[l]++, r = 0;
    for (int k = 0; k < l; ++k) r = uint16_t(r << 1 | ((c >> k) & 1));
    out[i] = r;
  }
  return out;
}

}  // namespace detail

namespace {

using detail::canonical_codes;
using detail::huffman_lengths;
using detail::Token;

int len_code(int len) {  // 3..258 -> 0..28
  int i = 28;
  while (kLenBase[i] > len) --i;
  return i;
}
int dist_code(int d) {  // 1..32768 -> 0..29
  int i = 29;
  while (kDistBase[i] > d) --i;
  return i;
}

}  // namespace

namespace detail {

// LZ77 with hash chains over a `window`-byte window and matches of up to
// `max_len` bytes. `effort` sets how far the chains are searched and whether
// matching is lazy; effort 0 finds nothing.
std::vector<Token> lz77(const uint8_t* in, size_t n, int effort, size_t window, size_t max_len) {
  std::vector<Token> t;
  t.reserve(n / 2 + 16);
  if (effort <= 0 || n < 4) {
    for (size_t i = 0; i < n; ++i) t.push_back({in[i], 0});
    return t;
  }
  static const int kChain[6] = {0, 8, 32, 64, 128, 1024};
  const int chain = kChain[std::min(effort, 5)];
  const bool lazy = effort >= 2;
  const size_t kWindow = window, kMax = max_len;
  constexpr int kBits = 15;
  std::vector<int64_t> head(size_t(1) << kBits, -1), prev(n, -1);
  auto hash3 = [&](size_t i) { return ((uint32_t(in[i]) << 10) ^ (uint32_t(in[i + 1]) << 5) ^ in[i + 2]) & ((1u << kBits) - 1); };
  auto insert = [&](size_t i) {
    if (i + 3 > n) return;
    const uint32_t h = hash3(i);
    prev[i] = head[h];
    head[h] = (int64_t)i;
  };
  auto best = [&](size_t i, size_t* out_dist) -> size_t {
    if (i + 3 > n) return 0;
    size_t best_len = 0;
    int64_t cand = head[hash3(i)];
    const size_t limit = std::min(kMax, n - i);
    for (int k = 0; cand >= 0 && k < chain; ++k, cand = prev[cand]) {
      const size_t d = i - (size_t)cand;
      if (d > kWindow) break;
      if (in[cand + best_len] != in[i + best_len]) continue;
      size_t l = 0;
      while (l < limit && in[cand + l] == in[i + l]) ++l;
      if (l > best_len) {
        best_len = l;
        *out_dist = d;
        if (l == limit) break;
      }
    }
    return best_len >= 3 ? best_len : 0;
  };
  size_t i = 0;
  while (i < n) {
    size_t d = 0;
    size_t l = best(i, &d);
    if (l && lazy && i + 1 < n) {
      size_t d2 = 0;
      insert(i);
      const size_t l2 = best(i + 1, &d2);
      if (l2 > l + 1) {
        t.push_back({in[i], 0});
        ++i;
        l = l2;
        d = d2;
      }
      if (l) {
        t.push_back({uint32_t(l), uint32_t(d)});
        for (size_t k = 1; k < l; ++k) insert(i + k);
        i += l;
        continue;
      }
    }
    if (l) {
      t.push_back({uint32_t(l), uint32_t(d)});
      for (size_t k = 0; k < l; ++k) insert(i + k);
      i += l;
    } else {
      t.push_back({in[i], 0});
      insert(i);
      ++i;
    }
  }
  return t;
}

}  // namespace detail

namespace {

using detail::lz77;

// One block, as whichever of stored, fixed and dynamic Huffman is smallest.
void deflate_block(BitOut& bo, const uint8_t* raw, size_t raw_n, const Token* t, size_t nt, bool last) {
  std::vector<uint32_t> lf(286, 0), df(30, 0);
  for (size_t i = 0; i < nt; ++i) {
    if (t[i].dist == 0) {
      lf[t[i].lit_or_len]++;
    } else {
      lf[257 + len_code(t[i].lit_or_len)]++;
      df[dist_code(t[i].dist)]++;
    }
  }
  lf[256] = 1;
  // Dynamic code lengths. Two codes at least in each alphabet keeps every
  // decoder's completeness checks happy.
  std::vector<uint8_t> ll = huffman_lengths(lf, 15);
  int used_d = 0;
  for (uint32_t f : df) used_d += f != 0;
  if (used_d < 2) {
    if (df[0] == 0) df[0] = 1;
    else df[1] = 1;
  }
  std::vector<uint8_t> dl = huffman_lengths(df, 15);
  int nlit = 286, ndist = 30;
  while (nlit > 257 && ll[nlit - 1] == 0) --nlit;
  while (ndist > 1 && dl[ndist - 1] == 0) --ndist;
  // The run-length coded lengths.
  std::vector<uint8_t> seq(ll.begin(), ll.begin() + nlit);
  seq.insert(seq.end(), dl.begin(), dl.begin() + ndist);
  std::vector<std::pair<uint8_t, uint8_t>> rle;  // (symbol, extra value)
  for (size_t i = 0; i < seq.size();) {
    size_t run = 1;
    while (i + run < seq.size() && seq[i + run] == seq[i]) ++run;
    if (seq[i] == 0 && run >= 3) {
      const size_t k = std::min<size_t>(run, 138);
      if (k >= 11) rle.push_back({18, uint8_t(k - 11)});
      else rle.push_back({17, uint8_t(k - 3)});
      i += k;
    } else if (seq[i] != 0 && run >= 4) {
      rle.push_back({seq[i], 0});
      const size_t k = std::min<size_t>(run - 1, 6);
      rle.push_back({16, uint8_t(k - 3)});
      i += 1 + k;
    } else {
      rle.push_back({seq[i], 0});
      ++i;
    }
  }
  std::vector<uint32_t> cf(19, 0);
  for (auto& r : rle) cf[r.first]++;
  std::vector<uint8_t> cl = huffman_lengths(cf, 7);
  int ncl = 19;
  while (ncl > 4 && cl[kClOrder[ncl - 1]] == 0) --ncl;
  // Sizes, in bits.
  static const std::vector<uint8_t> fixed_ll = [] {
    std::vector<uint8_t> l(288);
    for (int i = 0; i < 144; ++i) l[i] = 8;
    for (int i = 144; i < 256; ++i) l[i] = 9;
    for (int i = 256; i < 280; ++i) l[i] = 7;
    for (int i = 280; i < 288; ++i) l[i] = 8;
    return l;
  }();
  static const std::vector<uint8_t> fixed_dl(30, 5);
  auto body_bits = [&](const std::vector<uint8_t>& L, const std::vector<uint8_t>& D) {
    uint64_t b = L[256];
    for (size_t i = 0; i < nt; ++i) {
      if (t[i].dist == 0) {
        b += L[t[i].lit_or_len];
      } else {
        const int lc = len_code(t[i].lit_or_len), dc = dist_code(t[i].dist);
        b += L[257 + lc] + kLenExtra[lc] + D[dc] + kDistExtra[dc];
      }
    }
    return b;
  };
  uint64_t dyn = 3 + 5 + 5 + 4 + 3ull * ncl;
  for (auto& r : rle) dyn += cl[r.first] + (r.first == 16 ? 2 : r.first == 17 ? 3 : r.first == 18 ? 7 : 0);
  dyn += body_bits(ll, dl);
  const uint64_t fix = 3 + body_bits(fixed_ll, fixed_dl);
  const uint64_t stored = 3 + 7 + 32 + 8ull * raw_n + 40 * (raw_n / 65535);
  if (stored <= dyn && stored <= fix) {
    size_t off = 0;
    do {
      const size_t k = std::min<size_t>(raw_n - off, 65535);
      const bool fin = last && off + k == raw_n;
      bo.put(fin ? 1 : 0, 1);
      bo.put(0, 2);
      bo.flush();
      bo.o.push_back(uint8_t(k));
      bo.o.push_back(uint8_t(k >> 8));
      bo.o.push_back(uint8_t(~k));
      bo.o.push_back(uint8_t((~k) >> 8));
      bo.o.insert(bo.o.end(), raw + off, raw + off + k);
      off += k;
    } while (off < raw_n);
    return;
  }
  const bool use_fixed = fix <= dyn;
  const std::vector<uint8_t>& L = use_fixed ? fixed_ll : ll;
  const std::vector<uint8_t>& D = use_fixed ? fixed_dl : dl;
  bo.put(last ? 1 : 0, 1);
  bo.put(use_fixed ? 1 : 2, 2);
  if (!use_fixed) {
    bo.put(uint32_t(nlit - 257), 5);
    bo.put(uint32_t(ndist - 1), 5);
    bo.put(uint32_t(ncl - 4), 4);
    for (int i = 0; i < ncl; ++i) bo.put(cl[kClOrder[i]], 3);
    const std::vector<uint16_t> cc = canonical_codes(cl);
    for (auto& r : rle) {
      bo.put(cc[r.first], cl[r.first]);
      if (r.first == 16) bo.put(r.second, 2);
      else if (r.first == 17) bo.put(r.second, 3);
      else if (r.first == 18) bo.put(r.second, 7);
    }
  }
  const std::vector<uint16_t> lc = canonical_codes(L), dc = canonical_codes(D);
  for (size_t i = 0; i < nt; ++i) {
    if (t[i].dist == 0) {
      bo.put(lc[t[i].lit_or_len], L[t[i].lit_or_len]);
      continue;
    }
    const int lcode = len_code(t[i].lit_or_len), dcode = dist_code(t[i].dist);
    bo.put(lc[257 + lcode], L[257 + lcode]);
    bo.put(uint32_t(t[i].lit_or_len - kLenBase[lcode]), kLenExtra[lcode]);
    bo.put(dc[dcode], D[dcode]);
    bo.put(uint32_t(t[i].dist - kDistBase[dcode]), kDistExtra[dcode]);
  }
  bo.put(lc[256], L[256]);
}

}  // namespace

// No block is longer than its bytes stored: a block of 16 K tokens covers at
// least 16 KiB, and storing it costs a 5-byte header for each 64 KiB in it
// plus a byte of alignment.
size_t deflate_bound(size_t n) { return n + 6 * (n / 16384 + 1) + 5 * (n / 65535) + 8; }

Bytes deflate_compress(const uint8_t* in, size_t n, int effort) {
  Bytes o;
  o.reserve(deflate_bound(n) / 2);
  BitOut bo(o);
  const std::vector<Token> t = lz77(in, n, effort, 32768, 258);
  // Blocks of about 16 K tokens each, so the codes follow the data.
  constexpr size_t kTokens = 16384;
  size_t ti = 0, raw = 0;
  do {
    const size_t nt = std::min(kTokens, t.size() - ti);
    size_t raw_n = 0;
    for (size_t i = ti; i < ti + nt; ++i) raw_n += t[i].dist ? t[i].lit_or_len : 1;
    deflate_block(bo, in + raw, raw_n, t.data() + ti, nt, ti + nt == t.size());
    ti += nt;
    raw += raw_n;
  } while (ti < t.size());
  bo.flush();
  return o;
}

// ============================================================================
// gzip (RFC 1952)
// ============================================================================

// NVIDIA's bound: 74152 bytes for each 32 KiB, plus 24 (26 for no input),
// measured on an RTX 3060; far above what a stored DEFLATE stream needs.
size_t gzip_bound(size_t n) { return n == 0 ? 26 : 74152 * ((n + 32767) / 32768) + 24; }

Bytes gzip_compress(const uint8_t* in, size_t n, int effort) {
  // ID1 ID2 CM=8 FLG=0 MTIME=0 XFL=0 OS=255 (unknown), the header nvCOMP writes
  Bytes o = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff};
  const Bytes d = deflate_compress(in, n, effort);
  o.insert(o.end(), d.begin(), d.end());
  const uint32_t c = crc32(in, n), len = uint32_t(n);
  for (int i = 0; i < 4; ++i) o.push_back(uint8_t(c >> (8 * i)));
  for (int i = 0; i < 4; ++i) o.push_back(uint8_t(len >> (8 * i)));
  return o;
}

Result gunzip(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced) {
  *produced = 0;
  size_t ip = 0, total = 0;
  do {
    if (n - ip < 18 || in[ip] != 0x1f || in[ip + 1] != 0x8b || in[ip + 2] != 8) return Result::Corrupt;
    const uint8_t flg = in[ip + 3];
    if (flg & 0xE0) return Result::Corrupt;
    size_t p = ip + 10;
    if (flg & 4) {  // FEXTRA
      if (n - p < 2) return Result::Corrupt;
      const size_t xlen = rd16(in + p);
      if (n - p - 2 < xlen) return Result::Corrupt;
      p += 2 + xlen;
    }
    for (int f : {8, 16}) {  // FNAME, FCOMMENT: zero-terminated
      if (!(flg & f)) continue;
      while (p < n && in[p]) ++p;
      if (p >= n) return Result::Corrupt;
      ++p;
    }
    if (flg & 2) p += 2;  // FHCRC
    if (p > n) return Result::Corrupt;
    size_t got = 0, used = 0;
    uint8_t* dst = out ? out + total : nullptr;
    const Result r = inflate(in + p, n - p, dst, out ? cap - total : 0, &got, &used);
    if (r != Result::Ok) return r;
    p += used;
    if (n - p < 8) return Result::Corrupt;
    if (out && crc32(dst, got) != rd32(in + p)) return Result::BadChecksum;
    if (uint32_t(got) != rd32(in + p + 4)) return Result::Corrupt;
    total += got;
    ip = p + 8;
  } while (ip < n);
  *produced = total;
  return Result::Ok;
}

// ============================================================================
// Zstandard (RFC 8878)
// ============================================================================

namespace {

constexpr uint32_t kZstdMagic = 0xFD2FB528u;
constexpr size_t kZstdBlockMax = 128 * 1024;

// Literals_Length and Match_Length codes: baselines and extra bits (3.1.1.3.2.1).
constexpr uint32_t kLLBase[36] = {0,  1,  2,   3,   4,   5,    6,    7,    8,    9,     10,    11,
                                  12, 13, 14,  15,  16,  18,   20,   22,   24,   28,    32,    40,
                                  48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
constexpr uint8_t kLLBits[36] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  1,  1,
                                 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
constexpr uint32_t kMLBase[53] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  12,  13,   14,   15,   16,   17,    18,    19,   20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29,  30,  31,   32,   33,   34,   35,    37,    39,   41,
                                  43, 47, 51, 59, 67, 83, 99, 131, 259, 515, 1027, 2051, 4099, 8195, 16387, 32771, 65539};
constexpr uint8_t kMLBits[53] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

// The predefined distributions (3.1.1.3.2.2).
constexpr int16_t kLLDefault[36] = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2,  2,
                                    2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
constexpr int16_t kMLDefault[53] = {1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
constexpr int16_t kOFDefault[29] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
constexpr int kLLDefaultLog = 6, kMLDefaultLog = 6, kOFDefaultLog = 5;

// ---- the backward bitstream (4.1.1, 4.2.1.3) ----
//
// Written forward from the first byte, read from the last: the highest set bit
// of the last byte marks the end of the data. Reading past the beginning
// yields zeros, and is recorded.
struct BackBits {
  const uint8_t* p = nullptr;
  int64_t pos = 0;  // bits not yet read: the next read takes bits [pos - k, pos)
  bool init(const uint8_t* data, size_t n) {
    p = data;
    if (n == 0 || data[n - 1] == 0) return false;
    pos = int64_t(n) * 8 - (8 - highbit(data[n - 1]));
    return true;
  }
  uint64_t read(int k) {
    if (k == 0) return 0;
    uint64_t v = 0;
    for (int i = k - 1; i >= 0; --i) {
      const int64_t b = pos - k + i;
      const uint64_t bit = b >= 0 ? (p[b >> 3] >> (b & 7)) & 1 : 0;
      v |= bit << i;
    }
    pos -= k;
    return v;
  }
  uint64_t peek(int k) const {
    BackBits c = *this;
    return c.read(k);
  }
  bool overflowed() const { return pos < 0; }
  bool done() const { return pos == 0; }
};

// ---- FSE ----
struct FseEntry {
  uint16_t symbol;
  uint8_t bits;
  uint16_t base;
};

struct FseTable {
  int log = 0;
  std::vector<FseEntry> t;
};

// The decoding table from normalized counts (4.1.1).
bool fse_build(FseTable& ft, const int16_t* norm, int nsym, int log) {
  const uint32_t size = 1u << log;
  ft.log = log;
  ft.t.assign(size, FseEntry{0, 0, 0});
  std::vector<uint32_t> next(nsym);
  uint32_t high = size - 1;
  for (int s = 0; s < nsym; ++s) {
    if (norm[s] == -1) {
      ft.t[high--].symbol = (uint16_t)s;
      next[s] = 1;
    } else {
      next[s] = norm[s] > 0 ? (uint32_t)norm[s] : 0;
    }
  }
  const uint32_t step = (size >> 1) + (size >> 3) + 3, mask = size - 1;
  uint32_t pos = 0;
  for (int s = 0; s < nsym; ++s) {
    for (int i = 0; i < norm[s]; ++i) {
      ft.t[pos].symbol = (uint16_t)s;
      do pos = (pos + step) & mask;
      while (pos > high);
    }
  }
  if (pos != 0) return false;
  for (uint32_t u = 0; u < size; ++u) {
    const int s = ft.t[u].symbol;
    const uint32_t ns = next[s]++;
    if (ns == 0) return false;
    const int bits = log - highbit(ns);
    ft.t[u].bits = (uint8_t)bits;
    ft.t[u].base = uint16_t((ns << bits) - size);
  }
  return true;
}

// A forward LSB-first reader, for FSE table descriptions.
struct FwdBits {
  const uint8_t* p;
  size_t n;
  size_t pos = 0;
  uint32_t peek(int k) const {
    uint32_t v = 0;
    for (int i = 0; i < k; ++i) {
      const size_t b = pos + i;
      if ((b >> 3) < n) v |= uint32_t((p[b >> 3] >> (b & 7)) & 1) << i;
    }
    return v;
  }
  void skip(int k) { pos += k; }
};

// Reads an FSE table description (4.1.1); returns the bytes it used, or 0.
size_t fse_read(FseTable& ft, const uint8_t* p, size_t n, int max_sym, int max_log) {
  if (n == 0) return 0;
  FwdBits b{p, n};
  const int log = (int)b.peek(4) + 5;
  b.skip(4);
  if (log > max_log) return 0;
  int16_t norm[256] = {};
  int remaining = (1 << log) + 1, threshold = 1 << log, nbits = log + 1, sym = 0;
  bool prev0 = false;
  while (remaining > 1 && sym <= max_sym) {
    if (prev0) {
      int n0 = sym;
      while (b.peek(2) == 3) {
        n0 += 3;
        b.skip(2);
      }
      n0 += (int)b.peek(2);
      b.skip(2);
      if (n0 > max_sym) return 0;
      while (sym < n0) norm[sym++] = 0;
      if (sym > max_sym) break;
    }
    const int max = (2 * threshold - 1) - remaining;
    int count;
    if ((int)b.peek(nbits - 1) < max) {
      count = (int)b.peek(nbits - 1);
      b.skip(nbits - 1);
    } else {
      count = (int)b.peek(nbits);
      if (count >= threshold) count -= max;
      b.skip(nbits);
    }
    --count;
    remaining -= count < 0 ? -count : count;
    norm[sym++] = (int16_t)count;
    prev0 = count == 0;
    while (remaining < threshold) {
      --nbits;
      threshold >>= 1;
    }
    if ((b.pos >> 3) > n) return 0;
  }
  if (remaining != 1 || (b.pos + 7) / 8 > n) return 0;
  if (!fse_build(ft, norm, sym, log)) return 0;
  return (b.pos + 7) / 8;
}

struct FseState {
  const FseTable* t = nullptr;
  uint32_t s = 0;
  void init(const FseTable& table, BackBits& b) {
    t = &table;
    s = (uint32_t)b.read(table.log);
  }
  uint16_t symbol() const { return t->t[s].symbol; }
  void update(BackBits& b) {
    const FseEntry& e = t->t[s];
    s = e.base + (uint32_t)b.read(e.bits);
  }
};

// ---- Huffman for literals (4.2) ----
struct HufTable {
  int max_bits = 0;
  std::vector<uint8_t> sym, bits;  // indexed by the next max_bits bits
  bool valid() const { return max_bits > 0; }
};

bool huf_from_weights(HufTable& h, const std::vector<uint8_t>& w_in) {
  // w_in has the weights of all symbols but the last; the last is implied.
  if (w_in.empty() || w_in.size() > 255) return false;
  uint64_t total = 0;
  for (uint8_t w : w_in) {
    if (w > 11) return false;
    if (w) total += 1ull << (w - 1);
  }
  if (total == 0) return false;
  const int max_bits = highbit(total) + 1;
  if (max_bits > 11) return false;
  const uint64_t rest = (1ull << max_bits) - total;
  if (rest == 0 || (rest & (rest - 1))) return false;
  std::vector<uint8_t> w = w_in;
  w.push_back(uint8_t(highbit(rest) + 1));
  h.max_bits = max_bits;
  h.sym.assign(size_t(1) << max_bits, 0);
  h.bits.assign(size_t(1) << max_bits, 0);
  // Codes go to weights in increasing order, symbols in increasing order
  // within a weight: each symbol of weight w takes 2^(w-1) table slots.
  size_t at = 0;
  for (int wt = 1; wt <= max_bits; ++wt) {
    for (size_t s = 0; s < w.size(); ++s) {
      if (w[s] != wt) continue;
      const size_t k = size_t(1) << (wt - 1);
      if (at + k > h.sym.size()) return false;
      for (size_t i = 0; i < k; ++i) {
        h.sym[at + i] = (uint8_t)s;
        h.bits[at + i] = uint8_t(max_bits + 1 - wt);
      }
      at += k;
    }
  }
  return at == h.sym.size();
}

// Reads a Huffman tree description (4.2.1); returns the bytes used, or 0.
size_t huf_read(HufTable& h, const uint8_t* p, size_t n) {
  if (n == 0) return 0;
  const uint8_t hb = p[0];
  std::vector<uint8_t> w;
  if (hb >= 128) {  // direct: 4 bits per weight
    const size_t count = hb - 127, bytes = (count + 1) / 2;
    if (n < 1 + bytes) return 0;
    for (size_t i = 0; i < count; ++i) {
      const uint8_t b = p[1 + i / 2];
      w.push_back(i % 2 == 0 ? b >> 4 : b & 15);
    }
    return huf_from_weights(h, w) ? 1 + bytes : 0;
  }
  // FSE-compressed weights: two interleaved states over one table.
  const size_t csize = hb;
  if (csize == 0 || n < 1 + csize) return 0;
  FseTable ft;
  const size_t used = fse_read(ft, p + 1, csize, 255, 6);
  if (!used || used >= csize) return 0;
  BackBits b;
  if (!b.init(p + 1 + used, csize - used)) return 0;
  FseState s1, s2;
  s1.init(ft, b);
  s2.init(ft, b);
  if (b.overflowed()) return 0;
  for (;;) {
    w.push_back((uint8_t)s1.symbol());
    s1.update(b);
    if (b.overflowed()) {
      w.push_back((uint8_t)s2.symbol());
      break;
    }
    w.push_back((uint8_t)s2.symbol());
    s2.update(b);
    if (b.overflowed()) {
      w.push_back((uint8_t)s1.symbol());
      break;
    }
    if (w.size() > 255) return 0;
  }
  return huf_from_weights(h, w) ? 1 + csize : 0;
}

bool huf_stream(const HufTable& h, const uint8_t* p, size_t n, uint8_t* out, size_t count) {
  BackBits b;
  if (!b.init(p, n)) return false;
  for (size_t i = 0; i < count; ++i) {
    const uint32_t v = (uint32_t)b.peek(h.max_bits);
    out[i] = h.sym[v];
    b.read(h.bits[v]);
    if (b.overflowed()) return false;
  }
  return b.done();
}

struct ZstdFrameState {
  HufTable huf;
  FseTable ll, of, ml;
  bool ll_set = false, of_set = false, ml_set = false;
  uint64_t rep[3] = {1, 4, 8};
};

const FseTable& default_table(int which) {
  static const std::array<FseTable, 3> tables = [] {
    std::array<FseTable, 3> a;
    fse_build(a[0], kLLDefault, 36, kLLDefaultLog);
    fse_build(a[1], kOFDefault, 29, kOFDefaultLog);
    fse_build(a[2], kMLDefault, 53, kMLDefaultLog);
    return a;
  }();
  return tables[which];
}

// One of the three sequence tables, per its compression mode (3.1.1.3.2.1).
size_t seq_table(FseTable& slot, bool& set, int mode, int which, const uint8_t* p, size_t n, int max_sym,
                 int max_log, bool* ok) {
  *ok = true;
  switch (mode) {
    case 0:
      slot = default_table(which);
      set = true;
      return 0;
    case 1: {  // RLE: one symbol, no bits
      if (n < 1 || p[0] > max_sym) {
        *ok = false;
        return 0;
      }
      slot.log = 0;
      slot.t.assign(1, FseEntry{p[0], 0, 0});
      set = true;
      return 1;
    }
    case 2: {
      const size_t used = fse_read(slot, p, n, max_sym, max_log);
      if (!used) {
        *ok = false;
        return 0;
      }
      set = true;
      return used;
    }
    default:  // repeat the previous block's
      if (!set) *ok = false;
      return 0;
  }
}

Result zstd_block(ZstdFrameState& fs, const uint8_t* p, size_t n, Sink& s, size_t frame_start, size_t block_max) {
  if (n < 1) return Result::Corrupt;
  // ---- literals ----
  const int ltype = p[0] & 3, sf = (p[0] >> 2) & 3;
  size_t regen = 0, csize = 0, hdr = 0;
  int streams = 1;
  if (ltype <= 1) {
    if ((sf & 1) == 0) {
      hdr = 1;
      regen = p[0] >> 3;
    } else if (sf == 1) {
      if (n < 2) return Result::Corrupt;
      hdr = 2;
      regen = (p[0] >> 4) | (size_t(p[1]) << 4);
    } else {
      if (n < 3) return Result::Corrupt;
      hdr = 3;
      regen = (p[0] >> 4) | (size_t(p[1]) << 4) | (size_t(p[2]) << 12);
    }
    csize = ltype == 0 ? regen : 1;
  } else {
    const int sizes_bits[4] = {10, 10, 14, 18};
    hdr = sf <= 1 ? 3 : sf == 2 ? 4 : 5;
    streams = sf == 0 ? 1 : 4;
    if (n < hdr) return Result::Corrupt;
    uint64_t v = 0;
    for (size_t i = 0; i < hdr; ++i) v |= uint64_t(p[i]) << (8 * i);
    const int bits = sizes_bits[sf];
    regen = (v >> 4) & ((1ull << bits) - 1);
    csize = (v >> (4 + bits)) & ((1ull << bits) - 1);
  }
  if (regen > block_max || csize > n - hdr) return Result::Corrupt;
  std::vector<uint8_t> lit(regen);
  const uint8_t* lp = p + hdr;
  if (ltype == 0) {
    std::memcpy(lit.data(), lp, regen);
  } else if (ltype == 1) {
    std::memset(lit.data(), lp[0], regen);
  } else {
    size_t tree = 0;
    if (ltype == 2) {
      tree = huf_read(fs.huf, lp, csize);
      if (!tree) return Result::Corrupt;
    } else if (!fs.huf.valid()) {
      return Result::Corrupt;
    }
    const uint8_t* sp = lp + tree;
    const size_t sn = csize - tree;
    if (streams == 1) {
      if (!huf_stream(fs.huf, sp, sn, lit.data(), regen)) return Result::Corrupt;
    } else {
      if (sn < 6) return Result::Corrupt;
      const size_t s1 = rd16(sp), s2 = rd16(sp + 2), s3 = rd16(sp + 4);
      if (s1 + s2 + s3 > sn - 6) return Result::Corrupt;
      const size_t s4 = sn - 6 - s1 - s2 - s3, seg = (regen + 3) / 4;
      if (regen < 3 * seg) return Result::Corrupt;
      const uint8_t* q = sp + 6;
      const size_t sizes[4] = {s1, s2, s3, s4};
      const size_t counts[4] = {seg, seg, seg, regen - 3 * seg};
      size_t at = 0;
      for (int i = 0; i < 4; ++i) {
        if (!huf_stream(fs.huf, q, sizes[i], lit.data() + at, counts[i])) return Result::Corrupt;
        q += sizes[i];
        at += counts[i];
      }
    }
  }
  // ---- sequences ----
  const uint8_t* q = lp + csize;
  size_t qn = n - hdr - csize;
  if (qn < 1) return Result::Corrupt;
  size_t nseq = q[0], used = 1;
  if (nseq >= 128) {
    if (nseq < 255) {
      if (qn < 2) return Result::Corrupt;
      nseq = ((nseq - 128) << 8) + q[1];
      used = 2;
    } else {
      if (qn < 3) return Result::Corrupt;
      nseq = q[1] + (size_t(q[2]) << 8) + 0x7F00;
      used = 3;
    }
  }
  q += used;
  qn -= used;
  size_t lit_at = 0;
  if (nseq > 0) {
    if (qn < 1) return Result::Corrupt;
    const uint8_t modes = q[0];
    if (modes & 3) return Result::Corrupt;
    ++q;
    --qn;
    bool ok = true;
    size_t k = seq_table(fs.ll, fs.ll_set, modes >> 6, 0, q, qn, 35, 9, &ok);
    if (!ok) return Result::Corrupt;
    q += k;
    qn -= k;
    k = seq_table(fs.of, fs.of_set, (modes >> 4) & 3, 1, q, qn, 31, 8, &ok);
    if (!ok) return Result::Corrupt;
    q += k;
    qn -= k;
    k = seq_table(fs.ml, fs.ml_set, (modes >> 2) & 3, 2, q, qn, 52, 9, &ok);
    if (!ok) return Result::Corrupt;
    q += k;
    qn -= k;
    BackBits b;
    if (!b.init(q, qn)) return Result::Corrupt;
    FseState ll, of, ml;
    ll.init(fs.ll, b);
    of.init(fs.of, b);
    ml.init(fs.ml, b);
    for (size_t i = 0; i < nseq; ++i) {
      const int ofc = of.symbol(), mlc = ml.symbol(), llc = ll.symbol();
      if (ofc > 31 || mlc > 52 || llc > 35) return Result::Corrupt;
      const uint64_t ofv = (1ull << ofc) + b.read(ofc);
      const uint64_t mlv = kMLBase[mlc] + b.read(kMLBits[mlc]);
      const uint64_t llv = kLLBase[llc] + b.read(kLLBits[llc]);
      uint64_t off;
      if (ofv > 3) {
        off = ofv - 3;
        fs.rep[2] = fs.rep[1];
        fs.rep[1] = fs.rep[0];
        fs.rep[0] = off;
      } else {
        const uint64_t idx = ofv - 1 + (llv == 0 ? 1 : 0);  // 0, 1, 2, or 3 = rep[0] - 1
        if (idx == 0) {
          off = fs.rep[0];
        } else if (idx == 1) {
          off = fs.rep[1];
          fs.rep[1] = fs.rep[0];
          fs.rep[0] = off;
        } else if (idx == 2) {
          off = fs.rep[2];
          fs.rep[2] = fs.rep[1];
          fs.rep[1] = fs.rep[0];
          fs.rep[0] = off;
        } else {
          off = fs.rep[0] - 1;
          if (off == 0) return Result::Corrupt;
          fs.rep[2] = fs.rep[1];
          fs.rep[1] = fs.rep[0];
          fs.rep[0] = off;
        }
      }
      if (i + 1 < nseq) {
        ll.update(b);
        ml.update(b);
        of.update(b);
      }
      if (b.overflowed()) return Result::Corrupt;
      if (llv > regen - lit_at) return Result::Corrupt;
      if (!s.room(llv)) return Result::NoRoom;
      s.put(lit.data() + lit_at, llv);
      lit_at += llv;
      if (off > s.n - frame_start) return Result::Corrupt;
      if (!s.room(mlv)) return Result::NoRoom;
      s.copy(off, mlv);
    }
    if (!b.done()) return Result::Corrupt;
  } else if (qn != 0) {
    return Result::Corrupt;
  }
  if (!s.room(regen - lit_at)) return Result::NoRoom;
  s.put(lit.data() + lit_at, regen - lit_at);
  return Result::Ok;
}

}  // namespace

Result zstd_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced) {
  *produced = 0;
  Sink s{out, cap};
  size_t ip = 0;
  if (n == 0) return Result::Corrupt;
  while (ip < n) {
    if (n - ip < 4) return Result::Corrupt;
    const uint32_t magic = rd32(in + ip);
    if ((magic & 0xFFFFFFF0u) == 0x184D2A50u) {  // skippable frame
      if (n - ip < 8) return Result::Corrupt;
      const uint64_t len = rd32(in + ip + 4);
      if (len > n - ip - 8) return Result::Corrupt;
      ip += 8 + len;
      continue;
    }
    if (magic != kZstdMagic) return Result::Corrupt;
    ip += 4;
    if (ip >= n) return Result::Corrupt;
    const uint8_t fhd = in[ip++];
    const int fcs_flag = fhd >> 6, did_flag = fhd & 3;
    const bool single = (fhd >> 5) & 1, checksum = (fhd >> 2) & 1;
    if (fhd & 8) return Result::Corrupt;  // reserved bit
    uint64_t window = 0;
    if (!single) {
      if (ip >= n) return Result::Corrupt;
      const uint8_t wd = in[ip++];
      const int exp = wd >> 3, mant = wd & 7;
      const uint64_t base = 1ull << (10 + exp);
      window = base + (base / 8) * mant;
    }
    const int did_bytes[4] = {0, 1, 2, 4};
    if (n - ip < (size_t)did_bytes[did_flag]) return Result::Corrupt;
    uint32_t did = 0;
    for (int i = 0; i < did_bytes[did_flag]; ++i) did |= uint32_t(in[ip + i]) << (8 * i);
    ip += did_bytes[did_flag];
    if (did != 0) return Result::Corrupt;  // dictionaries: nvCOMP never makes them
    const int fcs_bytes = fcs_flag == 0 ? (single ? 1 : 0) : 1 << fcs_flag;
    if (n - ip < (size_t)fcs_bytes) return Result::Corrupt;
    uint64_t fcs = 0;
    bool has_fcs = fcs_bytes > 0;
    for (int i = 0; i < fcs_bytes; ++i) fcs |= uint64_t(in[ip + i]) << (8 * i);
    if (fcs_bytes == 2) fcs += 256;
    ip += fcs_bytes;
    if (single) window = fcs;
    const size_t block_max = (size_t)std::min<uint64_t>(std::max<uint64_t>(window, 1), kZstdBlockMax);
    const size_t frame_start = s.n;
    ZstdFrameState fs;
    bool last = false;
    while (!last) {
      if (n - ip < 3) return Result::Corrupt;
      const uint32_t bh = rd24(in + ip);
      ip += 3;
      last = bh & 1;
      const int btype = (bh >> 1) & 3;
      const size_t bsize = bh >> 3;
      if (btype == 3) return Result::Corrupt;
      if (btype == 1) {
        if (ip >= n) return Result::Corrupt;
        if (bsize > block_max) return Result::Corrupt;
        if (!s.room(bsize)) return Result::NoRoom;
        s.fill(in[ip], bsize);
        ip += 1;
        continue;
      }
      if (bsize > n - ip || bsize > block_max) return Result::Corrupt;
      if (btype == 0) {
        if (!s.room(bsize)) return Result::NoRoom;
        s.put(in + ip, bsize);
      } else {
        const Result r = zstd_block(fs, in + ip, bsize, s, frame_start, block_max);
        if (r != Result::Ok) return r;
      }
      ip += bsize;
    }
    if (has_fcs && s.n - frame_start != fcs) return Result::Corrupt;
    if (checksum) {
      if (n - ip < 4) return Result::Corrupt;
      if (out && uint32_t(xxh64(out + frame_start, s.n - frame_start)) != rd32(in + ip))
        return Result::BadChecksum;
      ip += 4;
    }
  }
  *produced = s.n;
  return Result::Ok;
}

// ---- encoding ----

namespace {

// An FSE encoder for one of the predefined distributions.
struct FseEnc {
  int log = 0;
  std::vector<uint16_t> state_table;
  struct Sym {
    int32_t delta_bits;
    int32_t delta_state;
  };
  std::vector<Sym> sym;
  void build(const int16_t* norm, int nsym, int lg) {
    log = lg;
    const uint32_t size = 1u << lg, mask = size - 1, step = (size >> 1) + (size >> 3) + 3;
    std::vector<uint16_t> spread(size);
    uint32_t high = size - 1;
    std::vector<uint32_t> cumul(nsym + 1, 0);
    for (int s = 0; s < nsym; ++s) {
      if (norm[s] == -1) {
        cumul[s + 1] = cumul[s] + 1;
        spread[high--] = (uint16_t)s;
      } else {
        cumul[s + 1] = cumul[s] + (uint32_t)std::max<int16_t>(norm[s], 0);
      }
    }
    uint32_t pos = 0;
    for (int s = 0; s < nsym; ++s)
      for (int i = 0; i < norm[s]; ++i) {
        spread[pos] = (uint16_t)s;
        do pos = (pos + step) & mask;
        while (pos > high);
      }
    state_table.assign(size, 0);
    std::vector<uint32_t> c(cumul.begin(), cumul.end());
    for (uint32_t u = 0; u < size; ++u) state_table[c[spread[u]]++] = uint16_t(size + u);
    sym.assign(nsym, Sym{0, 0});
    uint32_t total = 0;
    for (int s = 0; s < nsym; ++s) {
      const int cnt = norm[s];
      if (cnt == 0) {
        sym[s].delta_bits = ((lg + 1) << 16) - (int32_t)size;
      } else if (cnt == -1 || cnt == 1) {
        sym[s].delta_bits = (lg << 16) - (int32_t)size;
        sym[s].delta_state = (int32_t)total - 1;
        total += 1;
      } else {
        const int max_out = lg - highbit(uint32_t(cnt - 1));
        const uint32_t min_plus = uint32_t(cnt) << max_out;
        sym[s].delta_bits = (max_out << 16) - (int32_t)min_plus;
        sym[s].delta_state = (int32_t)total - cnt;
        total += (uint32_t)cnt;
      }
    }
  }
};

struct BitW {  // forward LSB-first, read back to front by the decoder
  Bytes o;
  uint64_t acc = 0;
  int n = 0;
  void put(uint64_t v, int k) {
    if (k == 0) return;
    v &= (k == 64) ? ~0ull : ((1ull << k) - 1);
    while (k > 0) {
      const int take = std::min(k, 32);
      acc |= (v & ((1ull << take) - 1)) << n;
      n += take;
      v >>= take;
      k -= take;
      while (n >= 8) {
        o.push_back(uint8_t(acc));
        acc >>= 8;
        n -= 8;
      }
    }
  }
  Bytes close() {  // the end mark: a 1 bit, then zeros to the byte
    put(1, 1);
    if (n > 0) o.push_back(uint8_t(acc));
    acc = 0;
    n = 0;
    return std::move(o);
  }
};

struct FseCState {
  uint32_t value = 0;
  const FseEnc* e = nullptr;
  void init(const FseEnc& enc, int s) {
    e = &enc;
    const FseEnc::Sym& t = enc.sym[s];
    const uint32_t nb = uint32_t((t.delta_bits + (1 << 15)) >> 16);
    const uint32_t v = (nb << 16) - (uint32_t)t.delta_bits;
    value = enc.state_table[(v >> nb) + t.delta_state];
  }
  void encode(BitW& w, int s) {
    const FseEnc::Sym& t = e->sym[s];
    const uint32_t nb = uint32_t((value + (uint32_t)t.delta_bits) >> 16);
    w.put(value, (int)nb);
    value = e->state_table[(value >> nb) + t.delta_state];
  }
  void flush(BitW& w) { w.put(value, e->log); }
};

const FseEnc& default_enc(int which) {
  static const std::array<FseEnc, 3> e = [] {
    std::array<FseEnc, 3> a;
    a[0].build(kLLDefault, 36, kLLDefaultLog);
    a[1].build(kOFDefault, 29, kOFDefaultLog);
    a[2].build(kMLDefault, 53, kMLDefaultLog);
    return a;
  }();
  return e[which];
}

int ll_code(uint32_t v) {
  if (v < 16) return (int)v;
  int c = 35;
  while (kLLBase[c] > v) --c;
  return c;
}
int ml_code(uint32_t mlen) {  // mlen >= 3
  if (mlen < 35) return (int)mlen - 3;
  int c = 52;
  while (kMLBase[c] > mlen) --c;
  return c;
}

struct Seq {
  uint32_t lit, mlen, offv;
};

// A compressed block: raw literals and the predefined sequence tables.
// Returns an empty vector when that would not be smaller than the block.
Bytes zstd_compressed_block(const uint8_t* src, size_t pos, size_t end, HashTable& ht) {
  std::vector<Seq> seqs;
  Bytes lits;
  size_t anchor = pos, ip = pos;
  constexpr size_t kMaxOffset = (size_t(1) << 28) - 4;
  while (ip + 8 <= end) {
    const uint32_t v = rd32(src + ip);
    int64_t& slot = ht.slot[ht.index(v)];
    const int64_t ref = slot;
    slot = (int64_t)ip;
    if (ref < 0 || ip - (size_t)ref > kMaxOffset || rd32(src + ref) != v) {
      ++ip;
      continue;
    }
    size_t len = 4;
    while (ip + len < end && src[ref + len] == src[ip + len] && len < 65536) ++len;
    lits.insert(lits.end(), src + anchor, src + ip);
    seqs.push_back({uint32_t(ip - anchor), uint32_t(len), uint32_t(ip - (size_t)ref + 3)});
    ip += len;
    anchor = ip;
  }
  lits.insert(lits.end(), src + anchor, src + end);
  Bytes o;
  const size_t ls = lits.size();
  if (ls < 32) {
    o.push_back(uint8_t(ls << 3));
  } else if (ls < 4096) {
    o.push_back(uint8_t(4 | ((ls & 15) << 4)));
    o.push_back(uint8_t(ls >> 4));
  } else {
    o.push_back(uint8_t(12 | ((ls & 15) << 4)));
    o.push_back(uint8_t(ls >> 4));
    o.push_back(uint8_t(ls >> 12));
  }
  o.insert(o.end(), lits.begin(), lits.end());
  const size_t ns = seqs.size();
  if (ns < 128) {
    o.push_back(uint8_t(ns));
  } else if (ns < 0x7F00) {
    o.push_back(uint8_t((ns >> 8) + 128));
    o.push_back(uint8_t(ns));
  } else {
    o.push_back(255);
    o.push_back(uint8_t(ns - 0x7F00));
    o.push_back(uint8_t((ns - 0x7F00) >> 8));
  }
  if (ns) {
    o.push_back(0);  // all three tables predefined
    const FseEnc &LL = default_enc(0), &OF = default_enc(1), &ML = default_enc(2);
    BitW w;
    auto codes = [](const Seq& s, int* llc, int* mlc, int* ofc) {
      *llc = ll_code(s.lit);
      *mlc = ml_code(s.mlen);
      *ofc = highbit(s.offv);
    };
    int llc, mlc, ofc;
    const Seq& lastq = seqs[ns - 1];
    codes(lastq, &llc, &mlc, &ofc);
    FseCState sml, sof, sll;
    sml.init(ML, mlc);
    sof.init(OF, ofc);
    sll.init(LL, llc);
    w.put(lastq.lit - kLLBase[llc], kLLBits[llc]);
    w.put(lastq.mlen - kMLBase[mlc], kMLBits[mlc]);
    w.put(lastq.offv - (1u << ofc), ofc);
    for (size_t i = ns - 1; i-- > 0;) {
      const Seq& q = seqs[i];
      codes(q, &llc, &mlc, &ofc);
      sof.encode(w, ofc);
      sml.encode(w, mlc);
      sll.encode(w, llc);
      w.put(q.lit - kLLBase[llc], kLLBits[llc]);
      w.put(q.mlen - kMLBase[mlc], kMLBits[mlc]);
      w.put(q.offv - (1u << ofc), ofc);
    }
    sml.flush(w);
    sof.flush(w);
    sll.flush(w);
    const Bytes bits = w.close();
    o.insert(o.end(), bits.begin(), bits.end());
  }
  if (o.size() >= end - pos) return Bytes();
  return o;
}

}  // namespace

// NVIDIA's bound, measured on an RTX 3060: a frame of 64 KiB blocks -- magic,
// descriptor, a window byte except for a single segment under 256 bytes, the
// content size, and three bytes a block. This encoder's frames (128 KiB
// blocks, single segment) never need more.
size_t zstd_bound(size_t n) {
  const size_t fcs = n < 256 ? 1 : n < 65792 ? 2 : n <= 0xFFFFFFFFull ? 4 : 8;
  const size_t blocks = std::max<size_t>(1, (n + 65535) / 65536);
  return n + 5 + (n < 256 ? 0 : 1) + fcs + 3 * blocks;
}

Bytes zstd_compress(const uint8_t* in, size_t n) {
  Bytes o;
  for (int i = 0; i < 4; ++i) o.push_back(uint8_t(kZstdMagic >> (8 * i)));
  // Single segment, the content size, no checksum, no dictionary.
  int fcs_flag;
  if (n < 256) fcs_flag = 0;
  else if (n < 65536 + 256) fcs_flag = 1;
  else if (n <= 0xFFFFFFFFull) fcs_flag = 2;
  else fcs_flag = 3;
  o.push_back(uint8_t(fcs_flag << 6 | 1 << 5));
  const uint64_t fcs = fcs_flag == 1 ? n - 256 : n;
  const int fcs_bytes = fcs_flag == 0 ? 1 : 1 << fcs_flag;
  for (int i = 0; i < fcs_bytes; ++i) o.push_back(uint8_t(fcs >> (8 * i)));
  HashTable ht(16);
  size_t pos = 0;
  do {
    const size_t end = std::min(n, pos + kZstdBlockMax);
    const size_t len = end - pos;
    const bool last = end == n;
    bool rle = len > 0;
    for (size_t i = pos + 1; rle && i < end; ++i) rle = in[i] == in[pos];
    Bytes body;
    int type;
    if (rle && len > 1) {
      type = 1;
      body.push_back(in[pos]);
    } else {
      body = len >= 16 ? zstd_compressed_block(in, pos, end, ht) : Bytes();
      if (body.empty()) {
        type = 0;
        body.assign(in + pos, in + end);
      } else {
        type = 2;
      }
    }
    // Block_Size: the content's size for a raw or RLE block, the compressed
    // size for a compressed one.
    const size_t bsize = type == 2 ? body.size() : len;
    const uint32_t bh = uint32_t(last) | uint32_t(type) << 1 | uint32_t(bsize) << 3;
    o.push_back(uint8_t(bh));
    o.push_back(uint8_t(bh >> 8));
    o.push_back(uint8_t(bh >> 16));
    o.insert(o.end(), body.begin(), body.end());
    pos = end;
  } while (pos < n);
  return o;
}

}  // namespace vgpu::codec
