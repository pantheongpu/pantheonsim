// The HEVC decoder and parser behind VirtualGPU's NVDEC (hevc_decode.cpp, hevc_parser.cpp), driven without
// the cuvid API. The streams in nvidia/tests/data/hevc were decoded by an RTX 3060's NVDEC and
// nvcuvid_hevc.rtx3060.txt holds the CRC-32 of every frame it displayed (NV12, or P016 for the 10-bit streams);
// the decoder here must produce the same frames in the same order, and must not fall over on damaged input.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/hevc_stream.hpp"
#include "../../src/hevc_syntax.hpp"
#include "vtest.hpp"

namespace {

std::string source_dir() {
  const char* e = std::getenv("VGPU_E2E_DATA");   // .../nvidia/tests/data, as the e2e scripts set it
  if (e && *e) return std::string(e) + "/../../..";
  return VGPU_SOURCE_DIR;
}

std::vector<uint8_t> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

uint32_t crc32(const uint8_t* d, size_t n, uint32_t crc = 0) {
  static uint32_t table[256];
  if (!table[1]) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
  }
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ d[i]) & 255] ^ (crc >> 8);
  return ~crc;
}

// The bytes the card's surface holds for the picture: NV12 rows (luma, then interleaved chroma), or P016 (the sample shifted up to 16 bits).
uint32_t frame_crc(const vgpu_hevc::OutFrame& f) {
  const bool wide = f.bit_depth_luma > 8;
  std::vector<uint8_t> out;
  auto put = [&](uint16_t v, int depth) {
    if (wide) {
      const unsigned s = static_cast<unsigned>(v) << (16 - depth);
      out.push_back(static_cast<uint8_t>(s & 255));
      out.push_back(static_cast<uint8_t>(s >> 8));
    } else {
      out.push_back(static_cast<uint8_t>(v));
    }
  };
  for (uint16_t v : f.y) put(v, f.bit_depth_luma);
  for (size_t i = 0; i < f.u.size(); ++i) {
    put(f.u[i], f.bit_depth_chroma);
    put(f.v[i], f.bit_depth_chroma);
  }
  return crc32(out.data(), out.size());
}

struct Expect {
  int w = 0, h = 0;
  uint32_t crc = 0;
};

// stream name -> frames the card displayed, from the "[packets]" sections of the transcript (the other sections play a stream
// another way: they show the same pictures, or have the card's scaler in them).
std::map<std::string, std::vector<Expect>> read_golden(const std::string& path) {
  std::map<std::string, std::vector<Expect>> out;
  std::ifstream f(path);
  std::string line, cur;
  bool keep = false;
  while (std::getline(f, line)) {
    if (line.rfind("stream ", 0) == 0) {
      std::istringstream ss(line.substr(7));
      std::string name, mode;
      ss >> name >> mode;
      keep = mode == "[packets]";
      cur = name;
      if (keep) out[cur];
      continue;
    }
    unsigned w, h, crc;
    int n;
    if (keep && std::sscanf(line.c_str(), " frame %d: %ux%u crc %x", &n, &w, &h, &crc) == 4) out[cur].push_back({static_cast<int>(w), static_cast<int>(h), crc});
  }
  return out;
}

uint32_t next_random(uint32_t& x) {
  x = x * 1103515245u + 12345u;
  return x >> 8;
}

}  // namespace

VTEST(every_stream_decodes_to_the_frames_the_cards_nvdec_displayed) {
  const std::string src = source_dir();
  const auto golden = read_golden(src + "/nvidia/tests/e2e/nvcuvid_hevc.rtx3060.txt");
  VCHECK(golden.size() > 30);
  int checked = 0;
  for (const auto& kv : golden) {
    const std::vector<uint8_t> data = slurp(src + "/nvidia/tests/data/hevc/" + kv.first + ".h265");
    VCHECK(!data.empty());
    std::vector<vgpu_hevc::OutFrame> frames;
    std::string err;
    const bool ok = vgpu_hevc::decode_stream(data.data(), data.size(), &frames, &err);
    if (!ok) std::fprintf(stderr, "%s: %s\n", kv.first.c_str(), err.c_str());
    VCHECK(ok);
    VCHECK_EQ(frames.size(), kv.second.size());
    for (size_t i = 0; i < frames.size() && i < kv.second.size(); ++i) {
      if (frames[i].width != kv.second[i].w || frames[i].height != kv.second[i].h || frame_crc(frames[i]) != kv.second[i].crc) {
        std::fprintf(stderr, "%s frame %zu: %dx%d crc %08x, the card's %dx%d crc %08x\n", kv.first.c_str(), i, frames[i].width, frames[i].height,
                     frame_crc(frames[i]), kv.second[i].w, kv.second[i].h, kv.second[i].crc);
        VCHECK(false);
        break;
      }
    }
    ++checked;
  }
  VCHECK(checked > 30);
}

VTEST(damaged_streams_do_not_crash_the_decoder) {
  const std::string src = source_dir();
  // streams that between them use every tool: tiles-free wavefronts, several slices, 10 bits, transform skip, lossless, weighted
  // prediction, scaling lists, long-term references, parameter sets between pictures, a resolution change
  for (const char* name : {"b_pyramid", "main10", "wpp", "slices", "tskip", "lossless", "cu_lossless", "weightb", "scaling_def", "amp_rect", "ctu16",
                           "b_ps_mid", "res_change", "cra_first", "open_gop", "crop_odd"}) {
    std::vector<uint8_t> data = slurp(src + "/nvidia/tests/data/hevc/" + name + ".h265");
    VCHECK(!data.empty());
    // truncated anywhere
    for (size_t cut : {size_t(0), size_t(7), data.size() / 7, data.size() / 3, data.size() / 2, data.size() - 5}) {
      std::vector<vgpu_hevc::OutFrame> frames;
      std::string err;
      (void)vgpu_hevc::decode_stream(data.data(), cut, &frames, &err);
    }
    // bits flipped at a deterministic scatter of places: in the picture data, and (a quarter of the rounds) in the first 200
    // bytes, where the parameter sets and the first slice header are
    uint32_t x = 12345;
    for (int round = 0; round < 40; ++round) {
      std::vector<uint8_t> d = data;
      for (int k = 0; k < 12; ++k) {
        const size_t span = round % 4 == 3 ? std::min<size_t>(200, d.size()) : d.size();
        const size_t at = next_random(x) % span;
        d[at] = static_cast<uint8_t>(d[at] ^ (1u << (next_random(x) & 7)));
      }
      std::vector<vgpu_hevc::OutFrame> frames;
      std::string err;
      (void)vgpu_hevc::decode_stream(d.data(), d.size(), &frames, &err);
    }
    // a run of bytes overwritten, and bytes removed
    for (int round = 0; round < 10; ++round) {
      std::vector<uint8_t> d = data;
      const size_t at = next_random(x) % d.size();
      const size_t len = std::min<size_t>(1 + next_random(x) % 40, d.size() - at);
      if (round & 1) {
        for (size_t i = 0; i < len; ++i) d[at + i] = static_cast<uint8_t>(next_random(x));
      } else {
        d.erase(d.begin() + static_cast<std::ptrdiff_t>(at), d.begin() + static_cast<std::ptrdiff_t>(at + len));
      }
      std::vector<vgpu_hevc::OutFrame> frames;
      std::string err;
      (void)vgpu_hevc::decode_stream(d.data(), d.size(), &frames, &err);
    }
  }
}

VTEST(random_bytes_behind_start_codes_do_not_crash_the_parser) {
  uint32_t x = 99;
  for (int round = 0; round < 200; ++round) {
    std::vector<uint8_t> d;
    const int nals = 1 + static_cast<int>(next_random(x) % 6);
    for (int n = 0; n < nals; ++n) {
      d.insert(d.end(), {0, 0, 1});
      // headers of the NAL unit types the parser looks at: VPS, SPS, PPS, IDR, TRAIL, SEI, AUD, EOS
      static const int types[] = {32, 33, 34, 19, 1, 39, 35, 36, 21, 8};
      d.push_back(static_cast<uint8_t>(types[next_random(x) % 10] << 1));
      d.push_back(1);
      const size_t len = next_random(x) % 120;
      for (size_t i = 0; i < len; ++i) d.push_back(static_cast<uint8_t>(next_random(x)));
    }
    std::vector<vgpu_hevc::OutFrame> frames;
    std::string err;
    (void)vgpu_hevc::decode_stream(d.data(), d.size(), &frames, &err);
  }
}

VTEST(exp_golomb_codes_read_back) {
  // 1 | 010 | 011 | 00100 | 00101 | 00110 | 00111 -> ue 0,1,2,3,4,5,6
  const uint8_t bytes[] = {0xA6, 0x42, 0x98, 0xE0};
  vgpu_hevc::BitReader b(bytes, sizeof bytes);
  for (uint32_t want = 0; want < 7; ++want) VCHECK_EQ(b.ue(), want);
  vgpu_hevc::BitReader s(bytes, sizeof bytes);
  const int want[] = {0, 1, -1, 2, -2, 3, -3};
  for (int w : want) VCHECK_EQ(s.se(), w);
}

VTEST_MAIN
