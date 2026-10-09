// See h264_stream.hpp.
#include "h264_stream.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>

#include "h264_decode.hpp"
#include "h264_parser.hpp"

namespace vgpu_h264 {
namespace {

class Collector : public ParserSink {
 public:
  explicit Collector(std::vector<OutFrame>* out) : out_(out) {}

  int sequence(const SeqInfo& i) override {
    seq_ = i;
    return std::max(i.min_surfaces, 1);
  }

  int decode(PicDesc& d) override {
    if (!failed_.empty()) return 1;
    PicParams pp = d.pp;
    std::unique_ptr<Frame>& cur = frames_[d.curr_pic_idx];
    if (!cur) cur = std::make_unique<Frame>();
    seq_of_[d.curr_pic_idx] = seq_;
    if (!d.second_field || cur->uid == 0) {
      cur->uid = ++uid_;
      cur->alloc(pp.mbs_w, pp.mbs_h);
    }
    pp.dpb.clear();
    for (const DpbSlot& s : d.slots) {
      if (s.pic_idx < 0) continue;
      if (s.used == 0 && !s.not_existing) continue;
      auto it = frames_.find(s.pic_idx);
      if (it == frames_.end() || !it->second || it->second->y.empty()) continue;
      DpbEntry e;
      e.frame = it->second.get();
      e.frame_num = s.frame_idx;
      e.long_term = s.is_long_term != 0;
      e.not_existing = s.not_existing != 0;
      e.used = s.used & 3;
      e.poc[0] = s.poc[0];
      e.poc[1] = s.poc[1];
      e.frame->poc[0] = s.poc[0];
      e.frame->poc[1] = s.poc[1];
      pp.dpb.push_back(e);
    }
    std::vector<SliceData> slices;
    for (const auto& nal : d.slice_nal) slices.push_back({d.data.data() + nal.first, nal.second});
    std::string err;
    if (!decode_picture(cur.get(), pp, slices, &err)) failed_ = err;
    return 1;
  }

  int display(const DisplayInfo& info) override {
    auto it = frames_.find(info.pic_idx);
    if (it == frames_.end() || !it->second) return 1;
    const Frame& f = *it->second;
    const SeqInfo seq_ = seq_of_[info.pic_idx];   // the sequence the frame was decoded in
    if (f.y.empty() || seq_.disp_right > f.stride_y || seq_.disp_bottom > f.mbs_h * 16) return 1;
    OutFrame o;
    const int l = seq_.disp_left & ~1, t = seq_.disp_top & ~1;
    o.width = seq_.disp_right - seq_.disp_left;
    o.height = seq_.disp_bottom - seq_.disp_top;
    o.timestamp = info.timestamp;
    o.y.resize(static_cast<size_t>(o.width) * o.height);
    for (int y = 0; y < o.height; ++y) std::memcpy(&o.y[static_cast<size_t>(y) * o.width], &f.y[static_cast<size_t>(t + y) * f.stride_y + l], static_cast<size_t>(o.width));
    const int cw = (o.width + 1) / 2, ch = (o.height + 1) / 2;
    o.uv.resize(static_cast<size_t>(cw) * ch * 2);
    for (int y = 0; y < ch; ++y)
      for (int x = 0; x < cw; ++x) {
        const size_t src = static_cast<size_t>(t / 2 + y) * f.stride_c + l / 2 + x;
        o.uv[(static_cast<size_t>(y) * cw + x) * 2] = f.u[src];
        o.uv[(static_cast<size_t>(y) * cw + x) * 2 + 1] = f.v[src];
      }
    out_->push_back(std::move(o));
    return 1;
  }

  std::string failed_;

 private:
  std::vector<OutFrame>* out_;
  SeqInfo seq_;
  std::map<int, SeqInfo> seq_of_;
  std::map<int, std::unique_ptr<Frame>> frames_;
  int uid_ = 0;
};

}  // namespace

bool decode_stream(const uint8_t* data, size_t n, std::vector<OutFrame>* out, std::string* error) {
  Collector sink(out);
  {
    H264Parser parser(&sink, 0, 10000000, 0, false);
    if (n) parser.feed(data, n, false, 0, false);
    parser.end_of_stream();
  }
  if (error) *error = sink.failed_;
  return sink.failed_.empty();
}

}  // namespace vgpu_h264
