// GDeflate (see nvcomp_gdeflate.hpp).
#include "nvcomp_gdeflate.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>

namespace vgpu::codec {
namespace {

using detail::Sink;

constexpr int kLanes = 32;

// DEFLATE's length and distance codes with the DEFLATE64 extensions (the
// draft's tables 1 and 2): code 285 carries 16 extra bits for any length
// from 3 to 65538, and distance codes 30 and 31 reach 64 KiB.
constexpr uint32_t kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                   31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 3};
constexpr uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                   2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 16};
constexpr uint32_t kDistBase[32] = {1,    2,    3,    4,    5,    7,     9,     13,    17,    25,   33,
                                    49,   65,   97,   129,  193,  257,   385,   513,   769,   1025, 1537,
                                    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 32769, 49153};
constexpr uint8_t kDistExtra[32] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4,  4,  5,  5,  6,  6,
                                    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14};
constexpr uint8_t kClOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

constexpr size_t kWindow = 65536, kMaxMatch = 65538;

// ============================================================================
// Decoding
// ============================================================================

// The 32 lanes' bit buffers, filled a word at a time from the interleaved
// stream. Bits past the end of the input read as zeros and are counted: a
// stream whose codes need them is truncated.
struct Lanes {
  const uint8_t* in;
  size_t n;
  size_t pos = 0;  // next word
  uint64_t buf[kLanes] = {};
  int cnt[kLanes] = {};
  int real[kLanes] = {};  // how many of cnt came from the input
  bool bad = false;

  void refill() {
    for (int l = 0; l < kLanes; ++l) {
      if (cnt[l] >= 32) continue;
      uint32_t w = 0;
      const size_t at = pos * 4;
      if (at < n) {
        for (size_t k = 0; k < 4 && at + k < n; ++k) w |= uint32_t(in[at + k]) << (8 * k);
        real[l] += 32;
      }
      ++pos;
      buf[l] |= uint64_t(w) << cnt[l];
      cnt[l] += 32;
    }
  }
  uint32_t bits(int l, int k) {
    if (k == 0) return 0;
    if (k > cnt[l] || k > real[l]) {
      bad = true;
      return 0;
    }
    const uint32_t v = uint32_t(buf[l] & ((1ull << k) - 1));
    buf[l] >>= k;
    cnt[l] -= k;
    real[l] -= k;
    return v;
  }
};

struct Huff {
  uint16_t count[16] = {};
  uint16_t symbol[320] = {};
  int nsym = 0;
  // Builds a canonical code from lengths; false for an over-subscribed set or
  // an incomplete one (other than a single code). A distance code may have no
  // codes at all -- a block without copies, as nvCOMP's entropy-only
  // algorithm writes -- and then any distance is corrupt.
  bool build(const uint8_t* len, int n, bool may_be_empty = false) {
    nsym = n;
    std::memset(count, 0, sizeof count);
    for (int i = 0; i < n; ++i) count[len[i]]++;
    if (count[0] == n) return may_be_empty;
    int left = 1;
    for (int l = 1; l < 16; ++l) {
      left = (left << 1) - count[l];
      if (left < 0) return false;
    }
    if (left > 0 && n - count[0] != 1) return false;
    uint16_t offs[16];
    offs[1] = 0;
    for (int l = 1; l < 15; ++l) offs[l + 1] = offs[l] + count[l];
    for (int i = 0; i < n; ++i)
      if (len[i]) symbol[offs[len[i]]++] = (uint16_t)i;
    return true;
  }
  int decode(Lanes& b, int l) const {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; ++len) {
      code |= (int)b.bits(l, 1);
      if (b.bad) return -1;
      const int c = count[len];
      if (code - c < first) return symbol[index + (code - first)];
      index += c;
      first += c;
      first <<= 1;
      code <<= 1;
    }
    return -1;
  }
};

// Output in symbol order: a copy's bytes are known only when its distance
// arrives, a round later, so symbols wait here until everything before them
// is resolved.
struct Pending {
  struct Sym {
    int lit;  // -1 for a copy
    uint32_t len;
    uint32_t dist;  // 0 until decoded
  };
  std::deque<Sym> q;
  size_t base = 0;  // index of q.front() among all symbols
  Result flush(Sink& s) {
    while (!q.empty()) {
      const Sym& y = q.front();
      if (y.lit >= 0) {
        if (!s.room(1)) return Result::NoRoom;
        s.put(uint8_t(y.lit));
      } else {
        if (y.dist == 0) break;
        if (y.dist > s.n) return Result::Corrupt;
        if (!s.room(y.len)) return Result::NoRoom;
        s.copy(y.dist, y.len);
      }
      q.pop_front();
      ++base;
    }
    return Result::Ok;
  }
};

Result decode_block_data(Lanes& b, Sink& s, const Huff& lh, const Huff& dh) {
  Pending p;
  size_t due[kLanes];  // symbol index of the copy waiting for this lane's distance
  bool pend[kLanes] = {};
  bool eob = false;
  auto distance = [&](int l) -> bool {
    const int ds = dh.decode(b, l);
    if (ds < 0 || ds >= dh.nsym || ds >= 32) return false;
    const uint32_t d = kDistBase[ds] + b.bits(l, kDistExtra[ds]);
    if (b.bad) return false;
    p.q[due[l] - p.base].dist = d;
    pend[l] = false;
    return true;
  };
  while (!eob) {
    b.refill();
    for (int l = 0; l < kLanes; ++l) {
      if (pend[l]) {
        if (!distance(l)) return Result::Corrupt;
        continue;
      }
      if (eob) continue;
      const int sym = lh.decode(b, l);
      if (sym < 0 || sym >= 286) return Result::Corrupt;
      if (sym < 256) {
        p.q.push_back({sym, 0, 0});
      } else if (sym == 256) {
        eob = true;
      } else {
        const int li = sym - 257;
        const uint32_t len = kLenBase[li] + b.bits(l, kLenExtra[li]);
        if (b.bad) return Result::Corrupt;
        due[l] = p.base + p.q.size();
        p.q.push_back({-1, len, 0});
        pend[l] = true;
      }
    }
    if (Result r = p.flush(s); r != Result::Ok) return r;
  }
  // Distances still due after the end-of-block round have one of their own.
  bool any = false;
  for (int l = 0; l < kLanes; ++l) any = any || pend[l];
  if (any) {
    b.refill();
    for (int l = 0; l < kLanes; ++l)
      if (pend[l] && !distance(l)) return Result::Corrupt;
  }
  if (Result r = p.flush(s); r != Result::Ok) return r;
  return p.q.empty() ? Result::Ok : Result::Corrupt;
}

}  // namespace

size_t gdeflate_bound(size_t n) { return (2 * n + 288) & ~size_t(3); }

Result gdeflate_decompress(const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced) {
  *produced = 0;
  if (n < 4 * kLanes) return Result::Corrupt;  // the first round reads 32 words
  Lanes b{in, n};
  Sink s{out, cap};
  bool last = false;
  while (!last) {
    // The block header: one round, lane 0 alone.
    b.refill();
    last = b.bits(0, 1);
    const uint32_t type = b.bits(0, 2);
    if (b.bad) return Result::Corrupt;
    Huff lh, dh;
    if (type == 0) {
      // Stored: a 16-bit length in a round of its own (no one's complement,
      // no byte alignment), then the bytes, one per lane per round.
      b.refill();
      const uint32_t len = b.bits(0, 16);
      if (b.bad) return Result::Corrupt;
      if (!s.room(len)) return Result::NoRoom;
      for (uint32_t done = 0; done < len;) {
        b.refill();
        for (int l = 0; l < kLanes && done < len; ++l, ++done) s.put(uint8_t(b.bits(l, 8)));
        if (b.bad) return Result::Corrupt;
      }
      continue;
    }
    if (type == 1) {
      uint8_t lens[288], d[32];
      for (int i = 0; i < 144; ++i) lens[i] = 8;
      for (int i = 144; i < 256; ++i) lens[i] = 9;
      for (int i = 256; i < 280; ++i) lens[i] = 7;
      for (int i = 280; i < 288; ++i) lens[i] = 8;
      std::fill(d, d + 32, 5);
      lh.build(lens, 288);
      dh.build(d, 32);
    } else if (type == 2) {
      b.refill();  // HLIT, HDIST, HCLEN: a round of lane 0
      const int nlit = (int)b.bits(0, 5) + 257, ndist = (int)b.bits(0, 5) + 1, ncl = (int)b.bits(0, 4) + 4;
      if (b.bad || nlit > 286) return Result::Corrupt;
      b.refill();  // the code length code's lengths: lane i holds the i-th
      uint8_t cl[19] = {};
      for (int i = 0; i < ncl; ++i) cl[kClOrder[i]] = (uint8_t)b.bits(i, 3);
      Huff ch;
      if (b.bad || !ch.build(cl, 19)) return Result::Corrupt;
      uint8_t lens[320] = {};
      const int total = nlit + ndist;
      for (int i = 0; i < total;) {
        b.refill();  // 32 code length symbols per round, lane order
        for (int l = 0; l < kLanes && i < total; ++l) {
          const int sym = ch.decode(b, l);
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
            rep = 3 + (int)b.bits(l, 2);
          } else if (sym == 17) {
            rep = 3 + (int)b.bits(l, 3);
          } else {
            rep = 11 + (int)b.bits(l, 7);
          }
          if (b.bad || i + rep > total) return Result::Corrupt;
          while (rep--) lens[i++] = v;
        }
      }
      if (lens[256] == 0 || !lh.build(lens, nlit) || !dh.build(lens + nlit, ndist, true)) return Result::Corrupt;
    } else {
      return Result::Corrupt;
    }
    if (Result r = decode_block_data(b, s, lh, dh); r != Result::Ok) return r;
  }
  *produced = s.n;
  return Result::Ok;
}

// ============================================================================
// Encoding
// ============================================================================

namespace {

struct Code {
  uint32_t bits = 0;  // LSB first: the (reversed) Huffman code, then its extra bits
  uint8_t len = 0;    // 0: the lane is idle this round
};
using Round = std::array<Code, kLanes>;

int len_code(uint32_t len) {  // 3..65538 -> 0..28
  if (len >= 258) return 28;
  int i = 27;
  while (kLenBase[i] > len) --i;
  return i;
}
int dist_code(uint32_t d) {  // 1..65536 -> 0..31
  int i = 31;
  while (kDistBase[i] > d) --i;
  return i;
}

// One dynamic-Huffman block, scheduled into rounds as the decoder reads it.
void schedule_block(std::vector<Round>& rounds, const std::vector<detail::Token>& t, bool last) {
  std::vector<uint32_t> lf(286, 0), df(32, 0);
  for (const detail::Token& k : t) {
    if (k.dist == 0) {
      lf[k.lit_or_len]++;
    } else {
      lf[257 + len_code(k.lit_or_len)]++;
      df[dist_code(k.dist)]++;
    }
  }
  lf[256] = 1;
  // At least two codes in each alphabet, so that every code is complete.
  auto two = [](std::vector<uint32_t>& f, size_t a, size_t b) {
    int used = 0;
    for (uint32_t x : f) used += x != 0;
    if (used >= 2) return;
    if (!f[a]) f[a] = 1;
    else f[b] = 1;
  };
  two(lf, 0, 1);
  two(df, 0, 1);
  const std::vector<uint8_t> ll = detail::huffman_lengths(lf, 15), dl = detail::huffman_lengths(df, 15);
  int nlit = 286, ndist = 32;
  while (nlit > 257 && ll[nlit - 1] == 0) --nlit;
  while (ndist > 1 && dl[ndist - 1] == 0) --ndist;
  std::vector<uint8_t> seq(ll.begin(), ll.begin() + nlit);
  seq.insert(seq.end(), dl.begin(), dl.begin() + ndist);
  std::vector<std::pair<uint8_t, uint8_t>> rle;
  for (size_t i = 0; i < seq.size();) {
    size_t run = 1;
    while (i + run < seq.size() && seq[i + run] == seq[i]) ++run;
    if (seq[i] == 0 && run >= 3) {
      const size_t k = std::min<size_t>(run, 138);
      rle.push_back(k >= 11 ? std::make_pair(uint8_t(18), uint8_t(k - 11)) : std::make_pair(uint8_t(17), uint8_t(k - 3)));
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
  two(cf, 0, 1);
  const std::vector<uint8_t> cl = detail::huffman_lengths(cf, 7);
  int ncl = 19;
  while (ncl > 4 && cl[kClOrder[ncl - 1]] == 0) --ncl;
  const std::vector<uint16_t> lc = detail::canonical_codes(ll), dc = detail::canonical_codes(dl),
                              cc = detail::canonical_codes(cl);

  Round r{};
  r[0] = {uint32_t(last) | 2u << 1, 3};
  rounds.push_back(r);
  r = Round{};
  r[0] = {uint32_t(nlit - 257) | uint32_t(ndist - 1) << 5 | uint32_t(ncl - 4) << 10, 14};
  rounds.push_back(r);
  r = Round{};
  for (int i = 0; i < ncl; ++i) r[i] = {cl[kClOrder[i]], 3};
  rounds.push_back(r);
  for (size_t i = 0; i < rle.size();) {
    r = Round{};
    for (int l = 0; l < kLanes && i < rle.size(); ++l, ++i) {
      const auto& e = rle[i];
      const int extra = e.first == 16 ? 2 : e.first == 17 ? 3 : e.first == 18 ? 7 : 0;
      r[l] = {uint32_t(cc[e.first]) | uint32_t(e.second) << cl[e.first], uint8_t(cl[e.first] + extra)};
    }
    rounds.push_back(r);
  }
  // The data: each free lane takes the next symbol; a copy's lane spends the
  // next round on its distance.
  Code due[kLanes];
  bool pend[kLanes] = {};
  size_t ti = 0;
  bool eob = false;
  while (!eob) {
    r = Round{};
    bool eob_here = false;
    for (int l = 0; l < kLanes; ++l) {
      if (pend[l]) {
        r[l] = due[l];
        pend[l] = false;
        continue;
      }
      if (eob_here) continue;
      if (ti < t.size()) {
        const detail::Token& k = t[ti++];
        if (k.dist == 0) {
          r[l] = {lc[k.lit_or_len], ll[k.lit_or_len]};
        } else {
          const int c = len_code(k.lit_or_len), d = dist_code(k.dist);
          r[l] = {uint32_t(lc[257 + c]) | (k.lit_or_len - kLenBase[c]) << ll[257 + c],
                  uint8_t(ll[257 + c] + kLenExtra[c])};
          due[l] = {uint32_t(dc[d]) | (k.dist - kDistBase[d]) << dl[d], uint8_t(dl[d] + kDistExtra[d])};
          pend[l] = true;
        }
      } else {
        r[l] = {lc[256], ll[256]};
        eob_here = eob = true;
      }
    }
    rounds.push_back(r);
  }
  bool any = false;
  for (int l = 0; l < kLanes; ++l) any = any || pend[l];
  if (any) {
    r = Round{};
    for (int l = 0; l < kLanes; ++l)
      if (pend[l]) r[l] = due[l];
    rounds.push_back(r);
  }
}

// A stored block: its length in a round of lane 0, then its bytes, a byte per
// lane per round.
void schedule_stored(std::vector<Round>& rounds, const uint8_t* raw, size_t n, bool last) {
  Round r{};
  r[0] = {uint32_t(last), 3};
  rounds.push_back(r);
  r = Round{};
  r[0] = {uint32_t(n), 16};
  rounds.push_back(r);
  for (size_t i = 0; i < n;) {
    r = Round{};
    for (int l = 0; l < kLanes && i < n; ++l, ++i) r[l] = {raw[i], 8};
    rounds.push_back(r);
  }
}

// A block with the fixed codes of RFC 1951 (five-bit distance codes, all 32
// of them usable here).
void schedule_fixed(std::vector<Round>& rounds, const std::vector<detail::Token>& t, bool last) {
  std::vector<uint8_t> ll(288), dl(32, 5);
  for (int i = 0; i < 144; ++i) ll[i] = 8;
  for (int i = 144; i < 256; ++i) ll[i] = 9;
  for (int i = 256; i < 280; ++i) ll[i] = 7;
  for (int i = 280; i < 288; ++i) ll[i] = 8;
  const std::vector<uint16_t> lc = detail::canonical_codes(ll), dc = detail::canonical_codes(dl);
  Round r{};
  r[0] = {uint32_t(last) | 1u << 1, 3};
  rounds.push_back(r);
  Code due[kLanes];
  bool pend[kLanes] = {};
  size_t ti = 0;
  bool eob = false;
  while (!eob) {
    r = Round{};
    bool eob_here = false;
    for (int l = 0; l < kLanes; ++l) {
      if (pend[l]) {
        r[l] = due[l];
        pend[l] = false;
        continue;
      }
      if (eob_here) continue;
      if (ti < t.size()) {
        const detail::Token& k = t[ti++];
        if (k.dist == 0) {
          r[l] = {lc[k.lit_or_len], ll[k.lit_or_len]};
        } else {
          const int c = len_code(k.lit_or_len), d = dist_code(k.dist);
          r[l] = {uint32_t(lc[257 + c]) | (k.lit_or_len - kLenBase[c]) << ll[257 + c],
                  uint8_t(ll[257 + c] + kLenExtra[c])};
          due[l] = {uint32_t(dc[d]) | (k.dist - kDistBase[d]) << 5, uint8_t(5 + kDistExtra[d])};
          pend[l] = true;
        }
      } else {
        r[l] = {lc[256], ll[256]};
        eob_here = eob = true;
      }
    }
    rounds.push_back(r);
  }
  bool any = false;
  for (int l = 0; l < kLanes; ++l) any = any || pend[l];
  if (any) {
    r = Round{};
    for (int l = 0; l < kLanes; ++l)
      if (pend[l]) r[l] = due[l];
    rounds.push_back(r);
  }
}

// Packs each lane's codes into its own words, then interleaves the words in
// the order the decoder's refills ask for them, ending with the refill after
// the last round.
Bytes interleave(const std::vector<Round>& rounds) {
  std::vector<uint32_t> words[kLanes];
  for (int l = 0; l < kLanes; ++l) {
    uint64_t acc = 0;
    int n = 0;
    for (const Round& r : rounds) {
      const Code& c = r[l];
      if (!c.len) continue;
      acc |= uint64_t(c.bits & ((1ull << c.len) - 1)) << n;
      n += c.len;
      if (n >= 32) {
        words[l].push_back(uint32_t(acc));
        acc >>= 32;
        n -= 32;
      }
    }
    if (n > 0) words[l].push_back(uint32_t(acc));
  }
  Bytes o;
  size_t next[kLanes] = {};
  int cnt[kLanes] = {};
  auto refill = [&] {
    for (int l = 0; l < kLanes; ++l) {
      if (cnt[l] >= 32) continue;
      const uint32_t w = next[l] < words[l].size() ? words[l][next[l]] : 0;
      ++next[l];
      for (int k = 0; k < 4; ++k) o.push_back(uint8_t(w >> (8 * k)));
      cnt[l] += 32;
    }
  };
  for (const Round& r : rounds) {
    refill();
    for (int l = 0; l < kLanes; ++l) cnt[l] -= r[l].len;
  }
  refill();
  return o;
}

}  // namespace

Bytes gdeflate_compress_blocks(const uint8_t* in, size_t n, int effort, const std::vector<int>& types) {
  const std::vector<detail::Token> t = detail::lz77(in, n, effort, kWindow, kMaxMatch);
  std::vector<Round> rounds;
  const size_t nb = std::max<size_t>(types.size(), 1);
  size_t ti = 0, raw = 0;
  for (size_t b = 0; b < nb; ++b) {
    const size_t end = b + 1 == nb ? t.size() : t.size() * (b + 1) / nb;
    const std::vector<detail::Token> part(t.begin() + ti, t.begin() + end);
    size_t raw_n = 0;
    for (const detail::Token& k : part) raw_n += k.dist ? k.lit_or_len : 1;
    const int type = types.empty() ? 2 : types[b];
    const bool last = b + 1 == nb;
    if (type == 0) schedule_stored(rounds, in + raw, raw_n, last);
    else if (type == 1) schedule_fixed(rounds, part, last);
    else schedule_block(rounds, part, last);
    ti = end;
    raw += raw_n;
  }
  return interleave(rounds);
}

Bytes gdeflate_compress(const uint8_t* in, size_t n, int effort) {
  const std::vector<detail::Token> t = detail::lz77(in, n, effort, kWindow, kMaxMatch);
  std::vector<Round> rounds;
  schedule_block(rounds, t, true);
  return interleave(rounds);
}

}  // namespace vgpu::codec
