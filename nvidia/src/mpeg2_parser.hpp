// The MPEG-2 half of VirtualGPU's video parser (cuvidCreateVideoParser): an elementary stream goes in; sequence, decode and display
// events come out, in the order and with the contents the parser of NVIDIA's libnvcuvid produces them -- the sequence format, the
// picture parameters of every picture (the reference pictures as indices of decode surfaces, the slice data with its offsets), picture
// indices handed out from a pool of decode surfaces, display order (a reference picture is shown when the next one has been decoded),
// the repeat count of the display, and timestamps. The rules were measured on an RTX 3060 (driver 595) with
// nvidia/tests/e2e/nvcuvid_mpeg2.cpp; where Rec. ITU-T H.262 leaves a choice (when a picture is recognised as complete, which
// pictures a decoder starting in the middle of a stream drops) the card's behaviour is followed.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mpeg2_decode.hpp"
#include "mpeg2_syntax.hpp"

namespace vgpu_mpeg2 {

struct SeqInfo {
  int coded_w = 0, coded_h = 0;
  int disp_left = 0, disp_top = 0, disp_right = 0, disp_bottom = 0;
  unsigned fps_num = 0, fps_den = 0;
  bool progressive = true;
  bool mpeg1 = false;         // no sequence extension: an MPEG-1 stream (the card names the codec MPEG-1 in the sequence callback)
  int chroma_format = 1;
  int min_surfaces = 4;
  unsigned bitrate = 0;
  int dar_x = 0, dar_y = 0;
  int video_format = 0, full_range = 0, primaries = 0, transfer = 0, matrix = 0;
  bool supported = true;      // false: a chroma format the decoder cannot handle (the sequence callback still runs)
};

struct PicDesc {
  int curr_pic_idx = 0;
  bool field_pic = false, bottom_field = false, second_field = false;
  bool ref_pic = true, intra_pic = false;
  int pic_width_in_mbs = 0;             // PicWidthInMbs as the card reports it (the width in samples divided by 16, rounded down)
  int fwd_idx = -1, bwd_idx = -1;       // ForwardRefIdx, BackwardRefIdx of CUVIDMPEG2PICPARAMS
  PicParams pp;                         // the decoder's view
  std::vector<uint8_t> data;            // the slices, each behind its start code
  std::vector<unsigned> slice_offsets;
  PicHeader hdr;
};

struct DisplayInfo {
  int pic_idx = 0;
  bool progressive_frame = true;
  bool top_field_first = false;
  int repeat_first_field = 0;
  int64_t timestamp = 0;
};

class ParserSink {
 public:
  virtual ~ParserSink() = default;
  // Returns the number of decode surfaces the application made (>1), or 1 / 0.
  virtual int sequence(const SeqInfo& info) = 0;
  virtual int decode(PicDesc& pic) = 0;
  virtual int display(const DisplayInfo& info) = 0;
};

class Mpeg2Parser {
 public:
  Mpeg2Parser(ParserSink* sink, unsigned max_decode_surfaces, unsigned clock_rate, unsigned max_display_delay);
  ~Mpeg2Parser();
  // `has_ts`: the packet carries a timestamp. `discontinuity`: bytes buffered so far are dropped first.
  // `end_of_picture`: the packet ends a picture (CUVID_PKT_ENDOFPICTURE).
  void feed(const uint8_t* data, size_t n, bool has_ts, int64_t ts, bool discontinuity, bool end_of_picture = false);
  void end_of_stream();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace vgpu_mpeg2
