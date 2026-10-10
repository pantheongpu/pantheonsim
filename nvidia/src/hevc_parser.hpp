// The HEVC half of VirtualGPU's video parser (cuvidCreateVideoParser): an Annex B byte stream goes in;
// sequence, decode and display events come out, in the order and with the contents the parser of
// NVIDIA's libnvcuvid produces them -- the sequence format, the picture parameters of every picture
// (the reference picture set as slots of a picture table, picture order counts, the slice data with its
// offsets), picture indices handed out from a pool of decode surfaces, display order by output bumping
// with the stream's reorder depth, the display delay, and timestamps. The rules were measured on an
// RTX 3060 (driver 595) with nvidia/tests/e2e/nvcuvid_hevc.cpp; where Rec. ITU-T H.265 leaves a choice
// (when a picture is recognised as complete, how many pictures the parser holds back) the card's
// behaviour is followed.
//
// Picture order counts, the reference picture set, marking and the output order DPB follow H.265
// clauses 8.3 and C.5.2.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "hevc_decode.hpp"
#include "hevc_syntax.hpp"

namespace vgpu_hevc {

struct SeqInfo {
  int coded_w = 0, coded_h = 0;
  int disp_left = 0, disp_top = 0, disp_right = 0, disp_bottom = 0;
  unsigned fps_num = 0, fps_den = 0;
  bool progressive = true;
  int chroma_format = 1, bit_depth_luma_minus8 = 0, bit_depth_chroma_minus8 = 0;
  int min_surfaces = 1;
  unsigned bitrate = 0;
  int dar_x = 0, dar_y = 0;
  int video_format = 5, full_range = 0, primaries = 2, transfer = 2, matrix = 2;
  int profile = 0, level = 0;
  bool supported = true;      // false: a profile or format the decoder cannot handle (the sequence callback still runs)
};

// One entry of the picture table NVDEC's picture parameters present: the pictures of the reference picture set.
struct RefSlot {
  int pic_idx = -1;
  int poc = 0;
  bool long_term = false;
};

struct PicDesc {
  int curr_pic_idx = 0;
  bool ref_pic = true, intra_pic = false;
  PicParams pp;               // the decoder's view; pp.refs[] holds only POCs and long-term marks (use `slots` for the pictures)
  RefSlot slots[16];
  std::vector<uint8_t> data;  // the slice segment NAL units, each behind a 3-byte start code
  std::vector<unsigned> slice_offsets;
  std::vector<std::pair<size_t, size_t>> slice_nal;   // (offset, size) of each slice's NAL unit (header first) within `data`
  int nal_type = 0;
  int num_delta_pocs_of_ref_rps = 0;
  const Sps* sps = nullptr;
  const Pps* pps = nullptr;
};

struct DisplayInfo {
  int pic_idx = 0;
  bool progressive_frame = true;
  bool top_field_first = false;
  int repeat_first_field = 0;
  int64_t timestamp = 0;
};

struct SeiMessage {
  int type = 0;
  std::vector<uint8_t> payload;
};

class ParserSink {
 public:
  virtual ~ParserSink() = default;
  // Returns the number of decode surfaces the application made (>1), or 1 / 0.
  virtual int sequence(const SeqInfo& info) = 0;
  virtual int decode(PicDesc& pic) = 0;
  virtual int display(const DisplayInfo& info) = 0;
  virtual void sei(int pic_idx, const std::vector<SeiMessage>& msgs) { (void)pic_idx; (void)msgs; }
};

class HevcParser {
 public:
  HevcParser(ParserSink* sink, unsigned max_decode_surfaces, unsigned clock_rate, unsigned max_display_delay, bool want_sei);
  ~HevcParser();
  // `has_ts`: the packet carries a timestamp. `discontinuity`: bytes buffered so far are dropped first.
  // `end_of_picture`: the packet ends a picture (CUVID_PKT_ENDOFPICTURE): its last NAL unit is complete.
  void feed(const uint8_t* data, size_t n, bool has_ts, int64_t ts, bool discontinuity, bool end_of_picture = false);
  void end_of_stream();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace vgpu_hevc
