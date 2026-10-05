// The host codecs behind libvgpunvcomp (nvidia/src/nvcomp_codecs.cpp,
// nvcomp_gdeflate.cpp), on their own: no simulator, no device memory.
//
// Every codec round-trips inputs of every kind and awkward size; every
// decoder reads the stored streams of NVIDIA's nvCOMP and of reference
// encoders (zstd, zlib, LZ4 HC, Snappy) back to their input, measures without
// writing, and reports a buffer one byte short; and corrupted streams are
// refused without reading or writing out of bounds (the sanitizer jobs run
// this). Interoperability with NVIDIA's own library in both directions is
// e2e_nvcomp_paths' job, on the card.
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "nvcomp_codecs.hpp"
#include "nvcomp_gdeflate.hpp"
#include "vtest.hpp"

#include "../e2e/nvcomp_vectors.inc"

using namespace vgpu::codec;

namespace {

enum class Codec { LZ4, Snappy, Deflate, Gzip, Zstd, Gdeflate };
const Codec kAll[] = {Codec::LZ4, Codec::Snappy, Codec::Deflate, Codec::Gzip, Codec::Zstd, Codec::Gdeflate};

Bytes encode(Codec c, const Bytes& in, int effort) {
  switch (c) {
    case Codec::LZ4: return lz4_compress(in.data(), in.size());
    case Codec::Snappy: return snappy_compress(in.data(), in.size());
    case Codec::Deflate: return deflate_compress(in.data(), in.size(), effort);
    case Codec::Gzip: return gzip_compress(in.data(), in.size(), effort);
    case Codec::Zstd: return zstd_compress(in.data(), in.size());
    case Codec::Gdeflate: return gdeflate_compress(in.data(), in.size(), effort);
  }
  return {};
}

Result decode(Codec c, const Bytes& in, uint8_t* out, size_t cap, size_t* produced) {
  switch (c) {
    case Codec::LZ4: return lz4_decompress(in.data(), in.size(), out, cap, produced);
    case Codec::Snappy: return snappy_decompress(in.data(), in.size(), out, cap, produced);
    case Codec::Deflate: return inflate(in.data(), in.size(), out, cap, produced);
    case Codec::Gzip: return gunzip(in.data(), in.size(), out, cap, produced);
    case Codec::Zstd: return zstd_decompress(in.data(), in.size(), out, cap, produced);
    case Codec::Gdeflate: return gdeflate_decompress(in.data(), in.size(), out, cap, produced);
  }
  return Result::Corrupt;
}

size_t bound(Codec c, size_t n) {
  switch (c) {
    case Codec::LZ4: return lz4_bound(n);
    case Codec::Snappy: return snappy_bound(n);
    case Codec::Deflate: return deflate_bound(n);
    case Codec::Gzip: return gzip_bound(n);
    case Codec::Zstd: return zstd_bound(n);
    case Codec::Gdeflate: return gdeflate_bound(n);
  }
  return 0;
}

Bytes sample(int kind, size_t n, unsigned seed) {
  std::mt19937 r(seed);
  Bytes v(n);
  for (size_t i = 0; i < n; ++i) {
    switch (kind) {
      case 0: v[i] = (uint8_t)r(); break;                                              // noise
      case 1: v[i] = (uint8_t)"a memory of the simulated device. "[i % 34]; break;   // periodic
      case 2: v[i] = 0; break;                                                         // one long run
      case 3: v[i] = (uint8_t)((r() % 4 == 0) ? r() % 3 : "abcdefgh"[(i / 7) % 8]); break;
      default: v[i] = (uint8_t)((i * i) >> 9); break;                                  // slowly varying
    }
  }
  return v;
}

// Decodes `s` as codec `c` and checks it gives `want`, measures it, and
// refuses a buffer one byte short.
void expect_decodes(Codec c, const Bytes& s, const Bytes& want) {
  Bytes out(want.size() + 1);
  size_t got = 0;
  VCHECK(decode(c, s, out.data(), want.size(), &got) == Result::Ok);
  VCHECK_EQ(got, want.size());
  out.resize(got);
  VCHECK(out == want);
  size_t measured = 0;
  VCHECK(decode(c, s, nullptr, 0, &measured) == Result::Ok);
  VCHECK_EQ(measured, want.size());
  if (!want.empty()) VCHECK(decode(c, s, out.data(), want.size() - 1, &got) == Result::NoRoom);
}

Bytes of(const unsigned char* p, size_t n) { return Bytes(p, p + n); }

}  // namespace

VTEST(round_trips_every_codec_size_and_effort) {
  const size_t sizes[] = {0, 1, 2, 3, 4, 5, 11, 12, 13, 31, 32, 33, 100, 255, 256, 257, 4096,
                          65535, 65536, 65537, 131072, 131073, 200000};
  for (Codec c : kAll)
    for (int kind = 0; kind < 5; ++kind)
      for (size_t n : sizes)
        for (int effort : {0, 1, 5}) {
          if (effort != 1 && c != Codec::Deflate && c != Codec::Gzip && c != Codec::Gdeflate) continue;
          const Bytes in = sample(kind, n, unsigned(n * 31 + kind));
          const Bytes s = encode(c, in, effort);
          VCHECK(s.size() <= bound(c, n));
          expect_decodes(c, s, in);
        }
}

VTEST(compression_actually_compresses) {
  const Bytes in = sample(1, 100000, 1);
  for (Codec c : kAll) VCHECK(encode(c, in, 1).size() < in.size() / 4);
  const Bytes zeros(70000, 0);
  for (Codec c : kAll) VCHECK(encode(c, zeros, 1).size() < 4000);
}

VTEST(decodes_nvidia_streams) {
  const Bytes in = nvcomp_vector_input();
  expect_decodes(Codec::LZ4, of(kNvLz4, sizeof kNvLz4), in);
  expect_decodes(Codec::Snappy, of(kNvSnappy, sizeof kNvSnappy), in);
  expect_decodes(Codec::Deflate, of(kNvDeflate, sizeof kNvDeflate), in);
  expect_decodes(Codec::Gzip, of(kNvGzip, sizeof kNvGzip), in);
  expect_decodes(Codec::Zstd, of(kNvZstd, sizeof kNvZstd), in);
  expect_decodes(Codec::Gdeflate, of(kNvGdeflate, sizeof kNvGdeflate), in);
  expect_decodes(Codec::Gdeflate, of(kNvGdeflate0, sizeof kNvGdeflate0), in);
  expect_decodes(Codec::Gdeflate, of(kNvGdeflate5, sizeof kNvGdeflate5), in);
}

VTEST(decodes_reference_encoder_streams) {
  const Bytes in = nvcomp_vector_input();
  expect_decodes(Codec::Zstd, of(kRefZstd19, sizeof kRefZstd19), in);
  expect_decodes(Codec::Zstd, of(kRefZstd1, sizeof kRefZstd1), in);
  expect_decodes(Codec::Deflate, of(kRefDeflate9, sizeof kRefDeflate9), in);
  expect_decodes(Codec::Gzip, of(kRefGzip6, sizeof kRefGzip6), in);
  expect_decodes(Codec::LZ4, of(kRefLz4Hc, sizeof kRefLz4Hc), in);
  expect_decodes(Codec::Snappy, of(kRefSnappy, sizeof kRefSnappy), in);
}

VTEST(our_stored_streams_still_decode) {
  const Bytes in = nvcomp_vector_input();
  expect_decodes(Codec::LZ4, of(kOurLz4, sizeof kOurLz4), in);
  expect_decodes(Codec::Snappy, of(kOurSnappy, sizeof kOurSnappy), in);
  expect_decodes(Codec::Deflate, of(kOurDeflate, sizeof kOurDeflate), in);
  expect_decodes(Codec::Deflate, of(kOurDeflate0, sizeof kOurDeflate0), in);
  expect_decodes(Codec::Gzip, of(kOurGzip, sizeof kOurGzip), in);
  expect_decodes(Codec::Zstd, of(kOurZstd, sizeof kOurZstd), in);
  expect_decodes(Codec::Gdeflate, of(kOurGdeflate, sizeof kOurGdeflate), in);
  expect_decodes(Codec::Gdeflate, of(kOurGdeflate0, sizeof kOurGdeflate0), in);
  for (const NvcompLayout& l : kGdeflateLayouts) expect_decodes(Codec::Gdeflate, of(l.data, l.size), in);
}

VTEST(gdeflate_block_layouts_round_trip) {
  const Bytes in = sample(3, 20000, 9);
  const std::vector<std::vector<int>> layouts = {{2}, {1}, {0}, {2, 2}, {0, 1, 2}, {1, 1, 1, 1}, {2, 0, 2, 0}};
  for (const auto& l : layouts) expect_decodes(Codec::Gdeflate, gdeflate_compress_blocks(in.data(), in.size(), 2, l), in);
  // NVIDIA's bound, which this encoder never exceeds.
  VCHECK_EQ(gdeflate_bound(0), 288u);
  VCHECK_EQ(gdeflate_bound(65537), 131360u);
  VCHECK_EQ(gdeflate_bound(100000), 200288u);
}

VTEST(corrupt_streams_are_refused_safely) {
  const Bytes in = nvcomp_vector_input();
  std::mt19937 r(3);
  for (Codec c : kAll) {
    const Bytes good = encode(c, in, 2);
    for (int k = 0; k < 400; ++k) {
      Bytes s = good;
      const int flips = 1 + int(r() % 4);
      for (int j = 0; j < flips; ++j) s[r() % s.size()] ^= uint8_t(1u << (r() % 8));
      if (k % 4 == 0) s.resize(r() % s.size());
      Bytes out(in.size());
      size_t got = 0;
      const Result res = decode(c, s, out.data(), out.size(), &got);
      // Whatever is decided, nothing was written past the buffer (the
      // sanitizers watch) and a success produced no more than fits.
      if (res == Result::Ok) VCHECK(got <= out.size());
      size_t m = 0;
      decode(c, s, nullptr, 0, &m);
    }
  }
  // Truncations of every length.
  for (Codec c : kAll) {
    const Bytes good = encode(c, in, 1);
    for (size_t n = 0; n < good.size(); n += 3) {
      Bytes s(good.begin(), good.begin() + n);
      Bytes out(in.size());
      size_t got = 0;
      const Result res = decode(c, s, out.data(), out.size(), &got);
      if (res == Result::Ok) VCHECK(got <= in.size());
    }
  }
}

VTEST(checksums) {
  const uint8_t msg[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  VCHECK_EQ(crc32(msg, 9), 0xCBF43926u);
  VCHECK_EQ(crc32(msg + 4, 5, crc32(msg, 4)), 0xCBF43926u);
  VCHECK_EQ(xxh64(nullptr, 0), 0xEF46DB3751D8E999ull);  // the specification's empty-input value
  // A gzip member whose CRC is wrong is reported as such.
  const Bytes in = nvcomp_vector_input();
  Bytes g = gzip_compress(in.data(), in.size(), 1);
  g[g.size() - 8] ^= 1;
  Bytes out(in.size());
  size_t got = 0;
  VCHECK(gunzip(g.data(), g.size(), out.data(), out.size(), &got) == Result::BadChecksum);
}

VTEST_MAIN
