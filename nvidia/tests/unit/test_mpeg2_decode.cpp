// The MPEG-2 decoder and parser behind VirtualGPU's NVDEC (mpeg2_decode.cpp, mpeg2_parser.cpp), driven without the cuvid API.
// The streams in nvidia/tests/data/mpeg2 were decoded by an RTX 3060's NVDEC and nvidia/tests/data/nvdec/mpeg2 holds the NV12
// frames it displayed. MPEG-2 leaves the inverse DCT to the decoder (H.262 Annex A bounds its error only), so the frames here are
// compared within the bounds of nvcuvid_mpeg2.cpp: at most 6 levels off, in at most a quarter of the samples. The decoder must
// also not fall over on damaged input.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../../src/mpeg2_stream.hpp"
#include "../../src/mpeg2_syntax.hpp"
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

uint32_t next_random(uint32_t& x) {
  x = x * 1103515245u + 12345u;
  return x >> 8;
}

constexpr int kMaxDiff = 6;
constexpr double kMaxDiffFraction = 0.25;

}  // namespace

VTEST(every_stream_decodes_to_the_frames_the_cards_nvdec_displayed) {
  const std::string src = source_dir();
  int checked = 0;
  size_t frames_checked = 0;
  std::vector<std::string> names;
  for (const auto& e : std::filesystem::directory_iterator(src + "/nvidia/tests/data/nvdec/mpeg2"))
    if (e.path().extension() == ".nv12") names.push_back(e.path().stem().string());
  std::sort(names.begin(), names.end());
  VCHECK(names.size() > 40);
  for (const std::string& name : names) {
    const std::vector<uint8_t> golden = slurp(src + "/nvidia/tests/data/nvdec/mpeg2/" + name + ".nv12");
    const std::vector<uint8_t> data = slurp(src + "/nvidia/tests/data/mpeg2/" + name + ".m2v");
    VCHECK(!data.empty());
    VCHECK(!golden.empty());
    std::vector<vgpu_mpeg2::OutFrame> frames;
    std::string err;
    const bool ok = vgpu_mpeg2::decode_stream(data.data(), data.size(), &frames, &err);
    if (!ok) std::fprintf(stderr, "%s: %s\n", name.c_str(), err.c_str());
    VCHECK(ok);
    size_t at = 0;
    bool same = true;
    for (size_t i = 0; i < frames.size() && same; ++i) {
      const std::vector<uint8_t> nv12 = vgpu_mpeg2::to_nv12(frames[i]);
      if (at + nv12.size() > golden.size()) {
        std::fprintf(stderr, "%s frame %zu: more frames than the card displayed\n", name.c_str(), i);
        same = false;
        break;
      }
      int maxd = 0;
      size_t nd = 0;
      for (size_t k = 0; k < nv12.size(); ++k) {
        const int dd = std::abs(static_cast<int>(nv12[k]) - static_cast<int>(golden[at + k]));
        if (dd) ++nd;
        maxd = std::max(maxd, dd);
      }
      if (maxd > kMaxDiff || static_cast<double>(nd) > kMaxDiffFraction * static_cast<double>(nv12.size())) {
        std::fprintf(stderr, "%s frame %zu: %dx%d, largest difference %d, %zu of %zu samples differ\n", name.c_str(), i, frames[i].width, frames[i].height, maxd, nd,
                     nv12.size());
        same = false;
      }
      at += nv12.size();
      ++frames_checked;
    }
    VCHECK(same);
    VCHECK_EQ(at, golden.size());   // the same number of frames, of the same sizes
    ++checked;
  }
  VCHECK(checked > 40);
  VCHECK(frames_checked > 200);
}

VTEST(display_delay_does_not_change_the_frames) {
  const std::string src = source_dir();
  for (const char* name : {"b_frames", "fields_ipb_tff", "pulldown"}) {
    const std::vector<uint8_t> data = slurp(src + "/nvidia/tests/data/mpeg2/" + name + ".m2v");
    VCHECK(!data.empty());
    std::vector<vgpu_mpeg2::OutFrame> base;
    std::string err;
    VCHECK(vgpu_mpeg2::decode_stream(data.data(), data.size(), &base, &err, 0));
    for (unsigned delay : {1u, 2u, 4u}) {
      std::vector<vgpu_mpeg2::OutFrame> frames;
      VCHECK(vgpu_mpeg2::decode_stream(data.data(), data.size(), &frames, &err, delay));
      VCHECK_EQ(frames.size(), base.size());
      for (size_t i = 0; i < frames.size() && i < base.size(); ++i) VCHECK(vgpu_mpeg2::to_nv12(frames[i]) == vgpu_mpeg2::to_nv12(base[i]));
    }
  }
}

VTEST(damaged_streams_do_not_crash_the_decoder) {
  const std::string src = source_dir();
  // streams that between them use every tool: frame and field pictures, B pictures, dual prime, concealment vectors, escapes, both
  // scans, downloaded matrices, several slices per row, a size change
  for (const char* name : {"b_frames", "fields_ipb_tff", "fields_mixed", "dual_prime_frames", "dual_prime_fields", "frames_escapes", "frames_slices",
                           "alt_scan", "matrices", "conceal", "progressive_tools", "pulldown", "seq_size_change", "frames_junk"}) {
    std::vector<uint8_t> data = slurp(src + "/nvidia/tests/data/mpeg2/" + name + ".m2v");
    VCHECK(!data.empty());
    // truncated anywhere
    for (size_t cut : {size_t(0), size_t(5), data.size() / 7, data.size() / 3, data.size() / 2, data.size() - 5}) {
      std::vector<vgpu_mpeg2::OutFrame> frames;
      std::string err;
      (void)vgpu_mpeg2::decode_stream(data.data(), cut, &frames, &err);
    }
    // bits flipped at a deterministic scatter of places: in the picture data, and (a quarter of the rounds) in the first 120
    // bytes, where the sequence header and the first picture headers are
    uint32_t x = 12345;
    for (int round = 0; round < 40; ++round) {
      std::vector<uint8_t> d = data;
      for (int k = 0; k < 12; ++k) {
        const size_t span = round % 4 == 3 ? std::min<size_t>(120, d.size()) : d.size();
        const size_t at = next_random(x) % span;
        d[at] = static_cast<uint8_t>(d[at] ^ (1u << (next_random(x) & 7)));
      }
      std::vector<vgpu_mpeg2::OutFrame> frames;
      std::string err;
      (void)vgpu_mpeg2::decode_stream(d.data(), d.size(), &frames, &err, round % 3);
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
      std::vector<vgpu_mpeg2::OutFrame> frames;
      std::string err;
      (void)vgpu_mpeg2::decode_stream(d.data(), d.size(), &frames, &err);
    }
  }
}

VTEST(random_bytes_behind_start_codes_do_not_crash_the_parser) {
  uint32_t x = 99;
  for (int round = 0; round < 300; ++round) {
    std::vector<uint8_t> d;
    const int units = 1 + static_cast<int>(next_random(x) % 8);
    for (int n = 0; n < units; ++n) {
      d.insert(d.end(), {0, 0, 1});
      // the start codes the parser looks at: picture, slices, user data, sequence header, extension, sequence end, group of pictures
      static const uint8_t codes[] = {0x00, 0x01, 0x02, 0x10, 0xAF, 0xB2, 0xB3, 0xB5, 0xB7, 0xB8};
      d.push_back(codes[next_random(x) % sizeof codes]);
      const size_t len = next_random(x) % 120;
      for (size_t i = 0; i < len; ++i) d.push_back(static_cast<uint8_t>(next_random(x)));
    }
    std::vector<vgpu_mpeg2::OutFrame> frames;
    std::string err;
    (void)vgpu_mpeg2::decode_stream(d.data(), d.size(), &frames, &err, round & 3);
  }
}

VTEST(random_pictures_behind_a_valid_sequence_do_not_crash_the_decoder) {
  // a sequence header and extension for 64x48 4:2:0, then picture headers with random fields and random slice data
  const uint8_t head[] = {0, 0, 1, 0xB3, 0x04, 0x00, 0x30, 0x14, 0xFF, 0xFF, 0xE0, 0x18,    // 64x48, 25 fps
                          0, 0, 1, 0xB5, 0x14, 0x8A, 0x00, 0x01, 0x00, 0x00};                 // sequence extension, progressive, 4:2:0
  uint32_t x = 7;
  for (int round = 0; round < 300; ++round) {
    std::vector<uint8_t> d(head, head + sizeof head);
    const int pics = 1 + static_cast<int>(next_random(x) % 5);
    for (int p = 0; p < pics; ++p) {
      d.insert(d.end(), {0, 0, 1, 0x00});
      d.push_back(static_cast<uint8_t>(next_random(x)));
      d.push_back(static_cast<uint8_t>((p == 0 ? 1 : 1 + next_random(x) % 3) << 3 | (next_random(x) & 7)));
      for (int i = 0; i < 3; ++i) d.push_back(static_cast<uint8_t>(next_random(x)));
      d.insert(d.end(), {0, 0, 1, 0xB5});   // picture coding extension
      for (int i = 0; i < 5; ++i) d.push_back(static_cast<uint8_t>(next_random(x)));
      const int slices = 1 + static_cast<int>(next_random(x) % 4);
      for (int s = 0; s < slices; ++s) {
        d.insert(d.end(), {0, 0, 1, static_cast<uint8_t>(1 + next_random(x) % 3)});
        const size_t len = next_random(x) % 160;
        for (size_t i = 0; i < len; ++i) {
          uint8_t b = static_cast<uint8_t>(next_random(x));
          if (b == 0 && i + 1 < len) b = 1;   // no accidental start codes
          d.push_back(b);
        }
      }
    }
    std::vector<vgpu_mpeg2::OutFrame> frames;
    std::string err;
    (void)vgpu_mpeg2::decode_stream(d.data(), d.size(), &frames, &err, round & 3);
  }
}

VTEST(sequence_header_fields_read_back) {
  // 720x576 (0x2D0 x 0x240), aspect code 2, frame rate code 3, 15000 x 400 bit/s, vbv buffer size 112, no matrices
  const uint8_t seq[] = {0x2D, 0x02, 0x40, 0x23, 0x0E, 0xA6, 0x23, 0x80};
  vgpu_mpeg2::SeqHeader h;
  VCHECK(vgpu_mpeg2::parse_sequence_header(seq, sizeof seq, &h));
  VCHECK_EQ(h.horizontal_size, 0x2D0);
  VCHECK_EQ(h.vertical_size, 0x240);
  VCHECK_EQ(h.aspect_ratio_information, 2);
  VCHECK_EQ(h.frame_rate_code, 3);
  VCHECK_EQ(h.bit_rate, 15000u);
  VCHECK_EQ(h.vbv_buffer_size, 112);
  VCHECK(!h.loaded_intra && !h.loaded_inter);
  vgpu_mpeg2::SeqHeader cut;
  VCHECK(!vgpu_mpeg2::parse_sequence_header(seq, 5, &cut));
}

VTEST(default_quantiser_matrices_are_table_7_6_and_flat) {
  uint8_t intra[64], inter[64];
  vgpu_mpeg2::default_matrices(intra, inter);
  VCHECK_EQ(intra[0], 8);
  VCHECK_EQ(intra[1], 16);
  VCHECK_EQ(intra[63], 83);
  VCHECK_EQ(intra[8], 16);   // raster order: the matrix is symmetric
  for (int i = 0; i < 64; ++i) VCHECK_EQ(inter[i], 16);
}

VTEST_MAIN
