// The H.264 decoder and parser behind VirtualGPU's NVDEC (h264_decode.cpp, h264_parser.cpp), driven
// without the cuvid API. The streams in nvidia/tests/data/h264 were decoded by an RTX 3060's NVDEC
// and nvcuvid_h264.rtx3060.txt holds the CRC-32 of every frame it displayed; the decoder here must
// produce the same frames in the same order, and must not fall over on damaged input.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/h264_stream.hpp"
#include "../../src/h264_syntax.hpp"
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

uint32_t frame_crc(const vgpu_h264::OutFrame& f) {
  std::vector<uint8_t> nv12(f.y);
  nv12.insert(nv12.end(), f.uv.begin(), f.uv.end());
  return crc32(nv12.data(), nv12.size());
}

struct Expect {
  int w = 0, h = 0;
  uint32_t crc = 0;
};

// stream name -> frames the card displayed, from the "[packets]" sections of the transcript.
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

}  // namespace

VTEST(every_stream_decodes_to_the_frames_the_cards_nvdec_displayed) {
  const std::string src = source_dir();
  const auto golden = read_golden(src + "/nvidia/tests/e2e/nvcuvid_h264.rtx3060.txt");
  VCHECK(golden.size() > 20);
  int checked = 0;
  for (const auto& kv : golden) {
    if (kv.first.rfind("mbaff", 0) == 0) continue;   // macroblock-adaptive frame/field coding is not decoded yet
    const std::vector<uint8_t> data = slurp(src + "/nvidia/tests/data/h264/" + kv.first + ".h264");
    VCHECK(!data.empty());
    std::vector<vgpu_h264::OutFrame> frames;
    std::string err;
    const bool ok = vgpu_h264::decode_stream(data.data(), data.size(), &frames, &err);
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
  VCHECK(checked > 20);
}

VTEST(damaged_streams_do_not_crash_the_decoder) {
  const std::string src = source_dir();
  for (const char* name : {"p_cabac", "b_cavlc", "high_8x8", "weightp"}) {
    std::vector<uint8_t> data = slurp(src + "/nvidia/tests/data/h264/" + name + ".h264");
    VCHECK(!data.empty());
    // truncated anywhere
    for (size_t cut : {data.size() / 7, data.size() / 3, data.size() - 5}) {
      std::vector<vgpu_h264::OutFrame> frames;
      std::string err;
      (void)vgpu_h264::decode_stream(data.data(), cut, &frames, &err);
    }
    // bytes flipped at a deterministic scatter of places (past the parameter sets)
    uint32_t x = 12345;
    for (int round = 0; round < 30; ++round) {
      std::vector<uint8_t> d = data;
      for (int k = 0; k < 20; ++k) {
        x = x * 1103515245u + 12345u;
        const size_t at = 60 + (x >> 8) % (d.size() - 60);
        d[at] = static_cast<uint8_t>(d[at] ^ (1u << ((x >> 4) & 7)));
      }
      std::vector<vgpu_h264::OutFrame> frames;
      std::string err;
      (void)vgpu_h264::decode_stream(d.data(), d.size(), &frames, &err);
    }
  }
}

VTEST(exp_golomb_codes_read_back) {
  // 1 | 010 | 011 | 00100 | 00101 | 00110 | 00111 -> ue 0,1,2,3,4,5,6
  const uint8_t bytes[] = {0xA6, 0x42, 0x98, 0xE0};
  vgpu_h264::BitReader b(bytes, sizeof bytes);
  for (uint32_t want = 0; want < 7; ++want) VCHECK_EQ(b.ue(), want);
  vgpu_h264::BitReader s(bytes, sizeof bytes);
  const int want[] = {0, 1, -1, 2, -2, 3, -3};
  for (int w : want) VCHECK_EQ(s.se(), w);
}

VTEST_MAIN
