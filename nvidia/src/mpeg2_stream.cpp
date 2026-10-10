// See mpeg2_stream.hpp.
#include "mpeg2_stream.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>

#include "mpeg2_decode.hpp"
#include "mpeg2_parser.hpp"

namespace vgpu_mpeg2 {
namespace {

class Collector : public ParserSink {
 public:
  explicit Collector(std::vector<OutFrame>* out) : out_(out) {}

  int sequence(const SeqInfo& i) override {
    seq_ = i;
    return std::max(i.min_surfaces, 1);
  }

  int decode(PicDesc& d) override {
    if (!seq_.supported) return 1;
    const PicParams& pp = d.pp;
    if (pp.mb_width <= 0 || pp.mb_height <= 0 || pp.mb_width > 255 || pp.mb_height > 255) return 1;
    std::unique_ptr<Frame>& cur = frames_[d.curr_pic_idx];
    if (!cur || !pp.second_field || cur->width != pp.mb_width * 16 || cur->height != pp.mb_height * 16) {
      cur = std::make_unique<Frame>();
      cur->alloc(pp.mb_width * 16, pp.mb_height * 16);
    }
    seq_of_[d.curr_pic_idx] = seq_;
    auto ref = [&](int idx) -> const Frame* {
      if (idx < 0 || idx == d.curr_pic_idx) return nullptr;
      auto it = frames_.find(idx);
      return it != frames_.end() && it->second && !it->second->y.empty() ? it->second.get() : nullptr;
    };
    std::vector<SliceData> slices;
    const size_t n = d.slice_offsets.size();
    for (size_t i = 0; i < n; ++i) {
      const size_t a = d.slice_offsets[i], b = i + 1 < n ? d.slice_offsets[i + 1] : d.data.size();
      if (a < b && b <= d.data.size()) slices.push_back({d.data.data() + a, b - a});
    }
    std::string err;
    if (!decode_picture(cur.get(), ref(d.fwd_idx), ref(d.bwd_idx), pp, slices, &err) && failed_.empty()) failed_ = err;
    return 1;
  }

  int display(const DisplayInfo& info) override {
    auto it = frames_.find(info.pic_idx);
    if (it == frames_.end() || !it->second) return 1;
    const Frame& f = *it->second;
    const SeqInfo seq = seq_of_[info.pic_idx];
    const int l = seq.disp_left, t = seq.disp_top;
    if (f.y.empty() || l < 0 || t < 0 || seq.disp_right <= l || seq.disp_bottom <= t || seq.disp_right > f.width || seq.disp_bottom > f.height) return 1;
    OutFrame o;
    o.width = seq.disp_right - l;
    o.height = seq.disp_bottom - t;
    o.timestamp = info.timestamp;
    o.y.resize(static_cast<size_t>(o.width) * static_cast<size_t>(o.height));
    for (int y = 0; y < o.height; ++y)
      std::memcpy(&o.y[static_cast<size_t>(y) * static_cast<size_t>(o.width)], &f.y[static_cast<size_t>(t + y) * static_cast<size_t>(f.stride_y) + static_cast<size_t>(l)],
                  static_cast<size_t>(o.width));
    const int cw = (o.width + 1) / 2, ch = (o.height + 1) / 2;
    o.u.resize(static_cast<size_t>(cw) * static_cast<size_t>(ch));
    o.v.resize(o.u.size());
    for (int y = 0; y < ch; ++y)
      for (int x = 0; x < cw; ++x) {
        const size_t src = static_cast<size_t>(t / 2 + y) * static_cast<size_t>(f.stride_c) + static_cast<size_t>(l / 2 + x);
        if (src >= f.u.size()) continue;
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

bool decode_stream(const uint8_t* data, size_t n, std::vector<OutFrame>* out, std::string* error, unsigned delay) {
  Collector sink(out);
  {
    Mpeg2Parser parser(&sink, 0, 10000000, delay);
    if (n) parser.feed(data, n, false, 0, false);
    parser.end_of_stream();
  }
  if (error) *error = sink.failed_;
  return sink.failed_.empty();
}

std::vector<uint8_t> to_nv12(const OutFrame& f) {
  const size_t w = static_cast<size_t>(f.width), h = static_cast<size_t>(f.height), cw = (w + 1) / 2, ch = (h + 1) / 2;
  std::vector<uint8_t> out(w * h + ch * ((w + 1) & ~size_t{1}), 0);
  std::memcpy(out.data(), f.y.data(), std::min(f.y.size(), w * h));
  uint8_t* uv = out.data() + w * h;
  for (size_t y = 0; y < ch; ++y)
    for (size_t x = 0; x < cw; ++x) {
      const size_t i = y * cw + x;
      uv[y * ((w + 1) & ~size_t{1}) + 2 * x] = i < f.u.size() ? f.u[i] : 0;
      uv[y * ((w + 1) & ~size_t{1}) + 2 * x + 1] = i < f.v.size() ? f.v[i] : 0;
    }
  return out;
}

}  // namespace vgpu_mpeg2
