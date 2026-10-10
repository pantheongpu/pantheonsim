// See hevc_stream.hpp.
#include "hevc_stream.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>

#include "hevc_decode.hpp"
#include "hevc_parser.hpp"

namespace vgpu_hevc {
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
    cur->alloc(pp.width, pp.height);
    cur->poc = pp.curr_poc;
    for (int i = 0; i < 16; ++i) {
      pp.refs[i] = RefPic();
      const RefSlot& s = d.slots[i];
      if (s.pic_idx < 0) continue;
      auto it = frames_.find(s.pic_idx);
      if (it == frames_.end() || !it->second || it->second->y.empty()) continue;
      pp.refs[i].frame = it->second.get();
      pp.refs[i].poc = s.poc;
      pp.refs[i].long_term = s.long_term;
      it->second->poc = s.poc;
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
    const SeqInfo seq = seq_of_[info.pic_idx];
    if (f.y.empty() || seq.disp_right > f.width || seq.disp_bottom > f.height) return 1;
    OutFrame o;
    const int l = seq.disp_left, t = seq.disp_top;
    o.width = seq.disp_right - seq.disp_left;
    o.height = seq.disp_bottom - seq.disp_top;
    o.bit_depth_luma = seq.bit_depth_luma_minus8 + 8;
    o.bit_depth_chroma = seq.bit_depth_chroma_minus8 + 8;
    o.timestamp = info.timestamp;
    o.y.resize(static_cast<size_t>(o.width) * static_cast<size_t>(o.height));
    for (int y = 0; y < o.height; ++y)
      std::memcpy(&o.y[static_cast<size_t>(y) * static_cast<size_t>(o.width)], &f.y[static_cast<size_t>(t + y) * static_cast<size_t>(f.stride_y) + static_cast<size_t>(l)],
                  sizeof(uint16_t) * static_cast<size_t>(o.width));
    const int cw = (o.width + 1) / 2, ch = (o.height + 1) / 2;
    o.u.resize(static_cast<size_t>(cw) * static_cast<size_t>(ch));
    o.v.resize(static_cast<size_t>(cw) * static_cast<size_t>(ch));
    for (int y = 0; y < ch; ++y)
      for (int x = 0; x < cw; ++x) {
        const size_t src = static_cast<size_t>(t / 2 + y) * static_cast<size_t>(f.stride_c) + static_cast<size_t>(l / 2 + x);
        o.u[static_cast<size_t>(y) * static_cast<size_t>(cw) + static_cast<size_t>(x)] = f.u[src];
        o.v[static_cast<size_t>(y) * static_cast<size_t>(cw) + static_cast<size_t>(x)] = f.v[src];
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
};

}  // namespace

bool decode_stream(const uint8_t* data, size_t n, std::vector<OutFrame>* out, std::string* error) {
  Collector sink(out);
  {
    HevcParser parser(&sink, 0, 10000000, 0, false);
    if (n) parser.feed(data, n, false, 0, false);
    parser.end_of_stream();
  }
  if (error) *error = sink.failed_;
  return sink.failed_.empty();
}

}  // namespace vgpu_hevc
