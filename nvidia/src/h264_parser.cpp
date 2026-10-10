// See h264_parser.hpp.
#include "h264_parser.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>

namespace vgpu_h264 {

namespace {

struct FrameStore {
  int pic_idx = -1;
  int frame_num = 0;
  int long_idx = -1;           // LongTermFrameIdx
  int ref_short = 0, ref_long = 0;   // bit 0 top field, bit 1 bottom field
  int poc[2] = {0, 0};
  int have = 0;                // fields decoded: bit 0 top, bit 1 bottom
  bool non_existing = false;
  bool need_output = false;
  bool progressive = true, tff = false;
  int repeat = 0;
  int slot = -1;
  bool in_dpb = false;
  bool queued = false;         // bumped, waiting in the display queue
  bool released = false;       // its surface went back to the pool
  bool current = false;        // the picture being decoded or stored
  int ref_any() const { return ref_short | ref_long; }
};
using FsPtr = std::shared_ptr<FrameStore>;

// Table A-1: MaxDpbMbs by level_idc (level 1b is level_idc 11 with constraint_set3, or 9).
long max_dpb_mbs(int level_idc, int constraint_flags) {
  switch (level_idc) {
    case 9: return 396;
    case 10: return 396;
    case 11: return (constraint_flags & 0x10) ? 396 : 900;
    case 12: case 13: case 20: return 2376;
    case 21: return 4752;
    case 22: case 30: return 8100;
    case 31: return 18000;
    case 32: return 20480;
    case 40: case 41: return 32768;
    case 42: return 34816;
    case 50: return 110400;
    case 51: case 52: return 184320;
    default: return 696320;
  }
}

int dpb_frames_of(const Sps& s) {
  // The intra profiles (High, High 10, High 4:2:2 and High 4:4:4 Predictive with constraint_set3_flag, CAVLC 4:4:4 Intra, ...) hold no
  // pictures back: max_dec_frame_buffering is inferred to be 0 (E.2.1) -- measured: the card asks for one decode surface.
  switch (s.profile_idc) {
    case 44: case 86: case 100: case 110: case 122: case 244:
      if (s.constraint_flags & 0x10) return 0;
      break;
    default: break;
  }
  const long per = max_dpb_mbs(s.level_idc, s.constraint_flags) / std::max(1, s.width_mbs * s.frame_height_mbs());
  return static_cast<int>(std::min<long>(per, 16));
}

int gcd_int(int a, int b) {
  while (b) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

struct ParamSet {
  std::vector<uint8_t> raw;     // RBSP, for detecting changes
  std::shared_ptr<Sps> sps;
  std::shared_ptr<Pps> pps;
};

}  // namespace

struct H264Parser::Impl {
  ParserSink* sink;
  unsigned max_surfaces, clock_rate, max_delay;
  bool want_sei;

  // ---- input ----
  std::vector<uint8_t> buf;     // bytes not yet consumed
  uint64_t buf_base = 0;        // stream offset of buf[0]
  uint64_t total_fed = 0;
  struct TsMark {
    uint64_t at;
    int64_t ts;
    bool used;
  };
  std::deque<TsMark> marks;
  bool peeked_this_nal = false;

  // ---- parameter sets ----
  ParamSet sps_raw[32], pps_raw[256];
  std::shared_ptr<Sps> sps_[32];
  std::shared_ptr<Pps> pps_[256];

  // ---- sequence ----
  std::shared_ptr<Sps> active_sps;
  SeqInfo last_info;
  bool have_info = false;
  int pool = 0;                 // decode surfaces
  std::deque<int> free_list;    // decode surfaces not in use, in the order they were released
  FsPtr held;                   // the picture displayed last
  int reorder_thr = 0;
  int dpb_size_ = 16;

  // ---- DPB ----
  std::vector<FsPtr> dpb;
  FsPtr slots[16];
  std::deque<FsPtr> out_queue;  // bumped, waiting for the display delay
  std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>> ts_heap;
  int64_t last_pic_ts = 0;
  bool have_last_ts = false;

  // ---- POC state ----
  int prev_poc_msb = 0, prev_poc_lsb = 0;
  int prev_frame_num_offset = 0, prev_frame_num = 0;
  int prev_ref_frame_num = 0;
  bool prev_had_mmco5 = false;
  int max_long_idx = -1;        // -1: no long-term frame indices

  // ---- the picture being assembled ----
  struct Cur {
    FsPtr fs;
    std::shared_ptr<Sps> sps;
    std::shared_ptr<Pps> pps;
    SliceHeader first, last;
    bool second_field = false;
    int structure = 0;          // 0 frame, 1 top field, 2 bottom field
    int poc[2] = {0, 0};
    int poc_msb = 0;
    int frame_num_offset = 0;
    PicDesc desc;
    bool all_intra = true;
    std::vector<uint8_t> nal_prefix;
    int pic_struct = -1;
    std::vector<SeiMessage> sei;
  };
  std::unique_ptr<Cur> cur;
  FsPtr pending_first_field;    // a first field stored, whose bumping waits for the second field
  int pending_pic_struct = -1;
  std::vector<SeiMessage> pending_sei;

  Impl(ParserSink* s, unsigned maxs, unsigned clock, unsigned delay, bool sei)
      : sink(s), max_surfaces(std::max(1u, maxs)), clock_rate(clock ? clock : 10000000), max_delay(delay), want_sei(sei) {}

  // ================================================================== input handling
  static int find_start(const std::vector<uint8_t>& b, size_t from) {
    for (size_t i = from; i + 3 <= b.size(); ++i)
      if (b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1) return static_cast<int>(i);
    return -1;
  }

  void feed(const uint8_t* d, size_t n, bool has_ts, int64_t ts, bool discontinuity, bool end_of_picture) {
    if (discontinuity) {
      finish_picture();
      buf.clear();
      buf_base = total_fed;
    }
    if (has_ts) marks.push_back({total_fed, ts, false});
    buf.insert(buf.end(), d, d + n);
    total_fed += n;
    process(false);
    if (end_of_picture) {
      // the packet holds whole pictures: its last NAL unit is complete, and so is the picture
      process(true);
      finish_picture();
    }
  }

  void end_of_stream() {
    process(true);
    finish_picture();
    flush_output(true);
  }

  void process(bool eos) {
    for (;;) {
      int s1 = find_start(buf, 0);
      if (s1 < 0) {
        // keep a possible partial start code at the end
        if (buf.size() > 2) {
          buf_base += buf.size() - 2;
          buf.erase(buf.begin(), buf.end() - 2);
        }
        return;
      }
      int sc_first = s1;
      if (sc_first > 0 && buf[sc_first - 1] == 0) --sc_first;   // the zero_byte of a four-byte start code
      const int s2 = find_start(buf, static_cast<size_t>(s1) + 3);
      if (s2 >= 0) {
        size_t end = static_cast<size_t>(s2);
        while (end > static_cast<size_t>(s1) + 3 && buf[end - 1] == 0) --end;
        consume_nal(buf.data() + s1 + 3, end - (s1 + 3), static_cast<size_t>(s2) - (s1 + 3), buf_base + static_cast<uint64_t>(sc_first));
        const size_t cut = static_cast<size_t>(s2) - ((s2 > 0 && buf[s2 - 1] == 0 && static_cast<size_t>(s2) - 1 >= static_cast<size_t>(s1) + 3) ? 1 : 0);
        buf_base += cut;
        buf.erase(buf.begin(), buf.begin() + static_cast<long>(cut));
        peeked_this_nal = false;
        continue;
      }
      // the last NAL unit is not terminated yet
      size_t end = buf.size();
      if (eos) {
        while (end > static_cast<size_t>(s1) + 3 && buf[end - 1] == 0) --end;
        if (end > static_cast<size_t>(s1) + 3) consume_nal(buf.data() + s1 + 3, end - (s1 + 3), end - (s1 + 3), buf_base + static_cast<uint64_t>(sc_first));
        buf_base += buf.size();
        buf.clear();
        return;
      }
      // NVIDIA's parser looks at a slice as soon as 256 bytes of it are buffered, to see whether it starts a new picture
      if (!peeked_this_nal && end - static_cast<size_t>(sc_first) >= 256) {
        const uint8_t* nal = buf.data() + s1 + 3;
        const int type = nal[0] & 31;
        if (type == 1 || type == 5) {
          peeked_this_nal = true;
          peek_slice(nal, end - (s1 + 3));
        }
      }
      return;
    }
  }

  void peek_slice(const uint8_t* nal, size_t n) {
    if (!cur) return;
    SliceHeader h;
    std::shared_ptr<Pps> pps;
    std::shared_ptr<Sps> sps;
    if (!parse_header(nal, n, &h, &pps, &sps)) return;
    if (is_new_picture(cur->last, h, *cur->sps)) finish_picture();
  }

  // ================================================================== NAL units
  bool parse_header(const uint8_t* nal, size_t n, SliceHeader* h, std::shared_ptr<Pps>* pps, std::shared_ptr<Sps>* sps) {
    if (n < 2) return false;
    const int ref_idc = nal[0] >> 5 & 3, type = nal[0] & 31;
    // only the first bytes are needed: unescape a prefix
    std::vector<uint8_t> rb = unescape(nal, std::min<size_t>(n, 128), 1);
    BitReader b(rb.data(), rb.size());
    // pps id is the third ue: peek to find the parameter sets
    BitReader p = b;
    p.ue();
    p.ue();
    const uint32_t pps_id = p.ue();
    if (pps_id > 255 || !pps_[pps_id]) return false;
    *pps = pps_[pps_id];
    *sps = sps_[(*pps)->sps_id];
    if (!*sps) return false;
    const HeaderCtx hc = make_ctx(**sps, **pps);
    // the whole header may need more than the prefix (long lists): parse from the full data to be safe
    std::vector<uint8_t> full = unescape(nal, n, 1);
    BitReader fb(full.data(), full.size());
    return parse_slice_header(fb, hc, ref_idc, type, h);
  }

  static HeaderCtx make_ctx(const Sps& s, const Pps& p) {
    HeaderCtx c;
    c.log2_max_frame_num = s.log2_max_frame_num;
    c.poc_type = s.poc_type;
    c.log2_max_poc_lsb = s.log2_max_poc_lsb;
    c.delta_pic_order_always_zero = s.delta_pic_order_always_zero;
    c.frame_mbs_only = s.frame_mbs_only;
    c.chroma_array_type = s.chroma_array_type();
    c.separate_colour_plane = s.separate_colour_plane;
    c.bottom_field_pic_order_in_frame_present = p.bottom_field_pic_order_in_frame_present;
    c.redundant_pic_cnt_present = p.redundant_pic_cnt_present;
    c.num_ref_idx_default[0] = p.num_ref_idx_default[0];
    c.num_ref_idx_default[1] = p.num_ref_idx_default[1];
    c.weighted_pred = p.weighted_pred;
    c.weighted_bipred_idc = p.weighted_bipred_idc;
    c.cabac = p.cabac;
    c.deblocking_control_present = p.deblocking_control_present;
    c.pic_init_qp = p.pic_init_qp;
    c.num_slice_groups = p.num_slice_groups;
    c.slice_group_map_type = p.slice_group_map_type;
    c.pic_size_in_map_units = s.width_mbs * s.height_map_units;
    c.slice_group_change_rate = p.slice_group_change_rate;
    return c;
  }

  void consume_nal(const uint8_t* nal, size_t n, size_t n_raw, uint64_t abs_offset) {
    if (n < 1) return;
    const int type = nal[0] & 31;
    switch (type) {
      case 1:
      case 5:
        consume_slice(nal, n, n_raw, abs_offset);
        break;
      // Parameter sets and SEI messages do not complete the picture before them -- the card's parser finishes a picture
      // only when it reads the first slice of the next one, or an access unit delimiter or end of sequence (measured with
      // streams that repeat or change the parameter sets and carry an SEI message before every slice).
      case 6:
        parse_sei(nal, n);
        break;
      case 7: {
        std::vector<uint8_t> rb = unescape(nal, n, 1);
        auto s = std::make_shared<Sps>();
        if (!parse_sps(rb.data(), rb.size(), s.get())) break;
        const int id = s->id;
        if (sps_raw[id].raw != rb || !sps_[id]) {
          sps_raw[id].raw = rb;
          sps_[id] = s;
        }
        break;
      }
      case 8: {
        std::vector<uint8_t> rb = unescape(nal, n, 1);
        auto p = std::make_shared<Pps>();
        std::vector<const Sps*> by_id(32, nullptr);
        for (int i = 0; i < 32; ++i) by_id[i] = sps_[i].get();
        if (!parse_pps(rb.data(), rb.size(), by_id.data(), p.get())) break;
        pps_raw[p->id].raw = rb;
        pps_[p->id] = p;
        break;
      }
      case 9:
      case 10:
      case 11:
        finish_picture();
        break;
      default:
        break;   // other NAL unit types (SVC / MVC extensions, auxiliary pictures) are skipped
    }
  }

  // ---- SEI (7.3.2.3, D.2) ----
  void parse_sei(const uint8_t* nal, size_t n) {
    std::vector<uint8_t> rb = unescape(nal, n, 1);
    size_t i = 0;
    while (i + 2 <= rb.size()) {
      if (i + 1 == rb.size() && rb[i] == 0x80) break;
      int type = 0, size = 0;
      while (i < rb.size() && rb[i] == 0xFF) {
        type += 255;
        ++i;
      }
      if (i >= rb.size()) break;
      type += rb[i++];
      while (i < rb.size() && rb[i] == 0xFF) {
        size += 255;
        ++i;
      }
      if (i >= rb.size()) break;
      size += rb[i++];
      if (i + static_cast<size_t>(size) > rb.size()) break;
      SeiMessage m;
      m.type = type;
      m.payload.assign(rb.begin() + static_cast<long>(i), rb.begin() + static_cast<long>(i) + size);
      if (type == 1 && active_or_any_sps()) {
        const Sps* s = active_or_any_sps();
        if (s->pic_struct_present) {
          BitReader b(m.payload.data(), m.payload.size());
          if (s->hrd_present) {
            b.u(s->cpb_removal_delay_len);
            b.u(s->dpb_output_delay_len);
          }
          pending_pic_struct = static_cast<int>(b.u(4));
        }
      }
      pending_sei.push_back(std::move(m));
      i += static_cast<size_t>(size);
    }
  }
  const Sps* active_or_any_sps() const {
    if (active_sps) return active_sps.get();
    for (const auto& s : sps_)
      if (s) return s.get();
    return nullptr;
  }

  // ================================================================== slices and pictures
  static bool is_new_picture(const SliceHeader& a, const SliceHeader& b, const Sps& sps) {
    if (a.frame_num != b.frame_num || a.pps_id != b.pps_id || a.field_pic != b.field_pic || a.bottom_field != b.bottom_field) return true;
    if ((a.nal_ref_idc == 0) != (b.nal_ref_idc == 0)) return true;
    if (sps.poc_type == 0 && (a.poc_lsb != b.poc_lsb || a.delta_poc_bottom != b.delta_poc_bottom)) return true;
    if (sps.poc_type == 1 && (a.delta_poc[0] != b.delta_poc[0] || a.delta_poc[1] != b.delta_poc[1])) return true;
    if (a.idr() != b.idr()) return true;
    if (a.idr() && b.idr() && a.idr_pic_id != b.idr_pic_id) return true;
    return false;
  }

  void consume_slice(const uint8_t* nal, size_t n, size_t n_raw, uint64_t abs_offset) {
    SliceHeader h;
    std::shared_ptr<Pps> pps;
    std::shared_ptr<Sps> sps;
    if (!parse_header(nal, n, &h, &pps, &sps)) return;
    // A slice that starts at macroblock 0 when the picture already has one is the next picture's, even when its header
    // says otherwise (consecutive IDR pictures with one idr_pic_id, which the Recommendation forbids and some encoders write).
    if (cur && (is_new_picture(cur->last, h, *cur->sps) || (h.first_mb == 0 && cur->first.first_mb == 0 && h.redundant_pic_cnt == 0 && !cur->desc.slice_offsets.empty())))
      finish_picture();
    if (!cur) {
      if (!begin_picture(h, pps, sps, abs_offset)) return;
    }
    cur->last = h;
    if (h.redundant_pic_cnt > 0) return;
    if (!h.is_intra()) cur->all_intra = false;
    PicDesc& d = cur->desc;
    d.slice_offsets.push_back(static_cast<unsigned>(d.data.size()));
    d.data.insert(d.data.end(), {0, 0, 1});
    d.slice_nal.push_back({d.data.size(), n});
    // the card's parser hands over the bytes up to the next start code, a zero_byte of a four-byte start code included
    d.data.insert(d.data.end(), nal, nal + n_raw);
  }

  // The sequence information for an SPS.
  SeqInfo make_info(const Sps& s) {
    SeqInfo i;
    i.coded_w = s.width_mbs * 16;
    i.coded_h = s.frame_height_mbs() * 16;
    const int cux = 2, cuy = 2 * (2 - s.frame_mbs_only);
    i.disp_left = s.crop[0] * cux;
    i.disp_right = i.coded_w - s.crop[1] * cux;
    i.disp_top = s.crop[2] * cuy;
    i.disp_bottom = i.coded_h - s.crop[3] * cuy;
    const bool crop_ok = s.crop[0] >= 0 && s.crop[1] >= 0 && s.crop[2] >= 0 && s.crop[3] >= 0 && s.crop[0] <= i.coded_w && s.crop[1] <= i.coded_w &&
                         s.crop[2] <= i.coded_h && s.crop[3] <= i.coded_h && i.disp_right > i.disp_left && i.disp_bottom > i.disp_top;
    if (!crop_ok) {   // a cropping rectangle outside the picture (not a conforming stream): the whole picture
      i.disp_left = i.disp_top = 0;
      i.disp_right = i.coded_w;
      i.disp_bottom = i.coded_h;
    }
    i.progressive = s.frame_mbs_only != 0;
    i.chroma_format = s.chroma_format_idc;
    i.bit_depth_luma_minus8 = s.bit_depth_luma - 8;
    i.bit_depth_chroma_minus8 = s.bit_depth_chroma - 8;
    i.profile = s.profile_idc;
    i.level = s.level_idc;
    int dpb = dpb_frames_of(s);
    if (s.restriction_present && s.max_dec_frame_buffering > 0) dpb = s.max_dec_frame_buffering;
    dpb = std::max(dpb, s.num_ref_frames);
    i.min_surfaces = dpb + 1;
    if (s.timing_present && s.num_units_in_tick && s.time_scale) {
      unsigned num = s.time_scale, den = 2 * s.num_units_in_tick;
      const unsigned g = static_cast<unsigned>(gcd_int(static_cast<int>(num), static_cast<int>(den)));
      i.fps_num = num / g;
      i.fps_den = den / g;
    }
    i.bitrate = s.hrd_bit_rate;
    // display aspect ratio: the sample aspect ratio times the display size
    int sar_w = 1, sar_h = 1;
    if (s.aspect_present) {
      static const int tab[17][2] = {{1, 1}, {1, 1}, {12, 11}, {10, 11}, {16, 11}, {40, 33}, {24, 11}, {20, 11}, {32, 11}, {80, 33}, {18, 11}, {15, 11}, {64, 33}, {160, 99}, {4, 3}, {3, 2}, {2, 1}};
      if (s.aspect_idc == 255) {
        sar_w = s.sar_w;
        sar_h = s.sar_h;
      } else if (s.aspect_idc >= 1 && s.aspect_idc <= 16) {
        sar_w = tab[s.aspect_idc][0];
        sar_h = tab[s.aspect_idc][1];
      }
      if (sar_w == 0 || sar_h == 0) sar_w = sar_h = 1;
    }
    long dx = static_cast<long>(i.disp_right - i.disp_left) * sar_w, dy = static_cast<long>(i.disp_bottom - i.disp_top) * sar_h;
    const long g = gcd_int(static_cast<int>(dx % 1000003), static_cast<int>(dy % 1000003)) ? 0 : 0;
    (void)g;
    long a = dx, b = dy;
    while (b) {
      const long t = a % b;
      a = b;
      b = t;
    }
    if (a > 0) {
      dx /= a;
      dy /= a;
    }
    i.dar_x = static_cast<int>(dx);
    i.dar_y = static_cast<int>(dy);
    if (s.video_signal_present) {
      i.video_format = s.video_format;
      i.full_range = s.video_full_range;
      if (s.colour_description_present) {
        i.primaries = s.colour_primaries;
        i.transfer = s.transfer;
        i.matrix = s.matrix;
      }
    }
    i.supported = s.chroma_format_idc == 1 && s.bit_depth_luma == 8 && s.bit_depth_chroma == 8 && !s.qpprime_bypass && !s.separate_colour_plane;
    return i;
  }

  static bool same_info(const SeqInfo& a, const SeqInfo& b) {
    return a.coded_w == b.coded_w && a.coded_h == b.coded_h && a.disp_left == b.disp_left && a.disp_top == b.disp_top && a.disp_right == b.disp_right &&
           a.disp_bottom == b.disp_bottom && a.fps_num == b.fps_num && a.fps_den == b.fps_den && a.progressive == b.progressive &&
           a.chroma_format == b.chroma_format && a.bit_depth_luma_minus8 == b.bit_depth_luma_minus8 && a.min_surfaces == b.min_surfaces &&
           a.bitrate == b.bitrate && a.dar_x == b.dar_x && a.dar_y == b.dar_y && a.video_format == b.video_format && a.full_range == b.full_range &&
           a.primaries == b.primaries && a.transfer == b.transfer && a.matrix == b.matrix;
  }

  // ---- surfaces ----
  // The pool is a queue of free surface indices: the card hands out the one that was released first.
  // A surface goes back to the pool when its picture is neither a reference nor waiting for output -- except the last
  // picture displayed: the application may still be reading it, so it stays out until the next picture is displayed
  // (or until the pool runs dry and it is taken back). Measured on the card.
  void try_free(const FsPtr& f) {
    if (f->released || f->pic_idx < 0 || f->ref_any() || f->need_output || f->queued || f->current || f == held || f == pending_first_field) return;
    f->released = true;
    free_list.push_back(f->pic_idx);
  }

  int alloc_pic_idx() {
    for (;;) {
      if (!free_list.empty()) {
        const int i = free_list.front();
        free_list.pop_front();
        return i;
      }
      if (held && !held->ref_any() && !held->need_output && !held->queued) {
        const int i = held->pic_idx;
        held->released = true;
        held.reset();
        return i;
      }
      // every surface is in use: show pictures until one is free
      if (!out_queue.empty()) {
        display_one();
      } else if (bump_one()) {
        display_one();
      } else {
        return -1;
      }
    }
  }

  // ---- output ----
  int waiting_count() const {
    int c = 0;
    for (const FsPtr& f : dpb)
      if (f->need_output) ++c;
    return c;
  }

  bool bump_one() {
    FsPtr best;
    for (const FsPtr& f : dpb)
      if (f->need_output && (!best || frame_poc(*f) < frame_poc(*best))) best = f;
    if (!best) return false;
    best->need_output = false;
    best->queued = true;
    out_queue.push_back(best);
    remove_unused();
    return true;
  }

  int frame_poc(const FrameStore& f) const { return f.have == 3 ? std::min(f.poc[0], f.poc[1]) : f.poc[f.have == 1 ? 0 : 1]; }

  void display_one() {
    FsPtr f = out_queue.front();
    out_queue.pop_front();
    f->queued = false;
    DisplayInfo di;
    di.pic_idx = f->pic_idx;
    di.progressive_frame = f->progressive;
    di.top_field_first = f->tff;
    di.repeat_first_field = f->repeat;
    di.timestamp = 0;
    if (!ts_heap.empty()) {
      di.timestamp = ts_heap.top();
      ts_heap.pop();
    }
    sink->display(di);
    FsPtr prev = held;
    held = f;
    if (prev) try_free(prev);
  }

  void pop_queue(size_t limit) {
    while (out_queue.size() > limit) display_one();
  }

  void remove_unused() {
    for (size_t i = 0; i < dpb.size();) {
      const FsPtr& f = dpb[i];
      if (!f->ref_any() && !f->need_output && f != pending_first_field && !(cur && cur->fs == f)) {
        if (f->slot >= 0) slots[f->slot].reset();
        f->slot = -1;
        f->in_dpb = false;
        dpb.erase(dpb.begin() + static_cast<long>(i));
      } else {
        ++i;
      }
    }
  }

  // The card's DPB array is the list of frame stores the DPB holds -- references and pictures waiting for output
  // alike: a picture takes the lowest free place when it is stored, and keeps it until it leaves the DPB.
  void assign_slot(const FsPtr& f) {
    if (f->slot >= 0) return;
    for (int i = 0; i < 16; ++i)
      if (!slots[i]) {
        slots[i] = f;
        f->slot = i;
        return;
      }
  }

  // Everything still waiting is shown (end of stream, a new sequence).
  void flush_output(bool all) {
    while (bump_one()) {
    }
    if (all) pop_queue(0);
  }

  // ---- picture order count (8.2.1) ----
  void compute_poc(Cur& c, const SliceHeader& h, const Sps& s) {
    const int max_fn = 1 << s.log2_max_frame_num;
    const bool idr = h.idr();
    int top = 0, bottom = 0;
    c.frame_num_offset = 0;
    if (s.poc_type == 0) {
      int pmsb = prev_poc_msb, plsb = prev_poc_lsb;
      if (idr) {
        pmsb = 0;
        plsb = 0;
      }
      const int max_lsb = 1 << s.log2_max_poc_lsb;
      int msb;
      if (h.poc_lsb < plsb && (plsb - h.poc_lsb) >= max_lsb / 2) msb = pmsb + max_lsb;
      else if (h.poc_lsb > plsb && (h.poc_lsb - plsb) > max_lsb / 2) msb = pmsb - max_lsb;
      else msb = pmsb;
      c.poc_msb = msb;
      if (!h.field_pic) {
        top = msb + h.poc_lsb;
        bottom = top + h.delta_poc_bottom;
      } else if (!h.bottom_field) {
        top = msb + h.poc_lsb;
      } else {
        bottom = msb + h.poc_lsb;
      }
    } else {
      int off;
      if (idr) off = 0;
      else if (prev_frame_num > h.frame_num) off = prev_frame_num_offset + max_fn;
      else off = prev_frame_num_offset;
      c.frame_num_offset = off;
      if (s.poc_type == 1) {
        const int n = static_cast<int>(s.offset_for_ref_frame.size());
        int abs_fn = n ? off + h.frame_num : 0;
        if (h.nal_ref_idc == 0 && abs_fn > 0) --abs_fn;
        int expected = 0;
        if (abs_fn > 0) {
          const int cycle = (abs_fn - 1) / n, in_cycle = (abs_fn - 1) % n;
          int per_cycle = 0;
          for (int v : s.offset_for_ref_frame) per_cycle += v;
          expected = cycle * per_cycle;
          for (int i = 0; i <= in_cycle; ++i) expected += s.offset_for_ref_frame[i];
        }
        if (h.nal_ref_idc == 0) expected += s.offset_for_non_ref_pic;
        if (!h.field_pic) {
          top = expected + h.delta_poc[0];
          bottom = top + s.offset_for_top_to_bottom + h.delta_poc[1];
        } else if (!h.bottom_field) {
          top = expected + h.delta_poc[0];
        } else {
          bottom = expected + s.offset_for_top_to_bottom + h.delta_poc[0];
        }
      } else {
        int temp;
        if (idr) temp = 0;
        else if (h.nal_ref_idc == 0) temp = 2 * (off + h.frame_num) - 1;
        else temp = 2 * (off + h.frame_num);
        top = bottom = temp;
      }
    }
    if (h.field_pic) {
      if (h.bottom_field) c.poc[1] = bottom;
      else c.poc[0] = top;
    } else {
      c.poc[0] = top;
      c.poc[1] = bottom;
    }
  }

  // ---- frame number gaps (8.2.5.2) ----
  void fill_frame_num_gap(const SliceHeader& h, const Sps& s) {
    const int max_fn = 1 << s.log2_max_frame_num;
    int unused = (prev_ref_frame_num + 1) % max_fn;
    int guard = 0;
    while (unused != h.frame_num && guard++ < 64) {
      auto f = std::make_shared<FrameStore>();
      f->non_existing = true;
      f->frame_num = unused;
      f->have = 3;
      f->ref_short = 3;
      f->need_output = false;
      // sliding window
      sliding_window(unused, s, nullptr);
      // order counts of a non-existing frame for POC types 1 and 2
      if (s.poc_type != 0) {
        SliceHeader fake;
        fake.frame_num = unused;
        fake.nal_ref_idc = 1;
        Cur tmp;
        compute_poc(tmp, fake, s);
        f->poc[0] = tmp.poc[0];
        f->poc[1] = tmp.poc[1];
        prev_frame_num_offset = tmp.frame_num_offset;
        prev_frame_num = unused;
      }
      f->in_dpb = true;
      dpb.push_back(f);
      remove_unused();
      assign_slot(f);
      prev_ref_frame_num = unused;
      unused = (unused + 1) % max_fn;
    }
  }

  // ---- reference marking (8.2.5) ----
  static int wrap_of(const FrameStore& f, int cur_fn, int max_fn) { return f.frame_num > cur_fn ? f.frame_num - max_fn : f.frame_num; }

  void sliding_window(int cur_fn, const Sps& s, const FrameStore* cur_fs) {
    const int max_fn = 1 << s.log2_max_frame_num;
    int num_short = 0, num_long = 0;
    for (const FsPtr& f : dpb) {
      if (f.get() == cur_fs) continue;
      if (f->ref_short) ++num_short;
      else if (f->ref_long) ++num_long;
    }
    if (num_short + num_long >= std::max(s.num_ref_frames, 1) && num_short > 0) {
      FsPtr victim;
      int best = 0;
      for (const FsPtr& f : dpb) {
        if (f.get() == cur_fs || !f->ref_short) continue;
        const int w = wrap_of(*f, cur_fn, max_fn);
        if (!victim || w < best) {
          victim = f;
          best = w;
        }
      }
      if (victim) {
        victim->ref_short = 0;
        try_free(victim);
      }
    }
  }

  void mark_current(Cur& c) {
    FrameStore& fs = *c.fs;
    const Sps& s = *c.sps;
    const SliceHeader& h = c.first;
    const int mask = c.structure == 0 ? 3 : c.structure;
    const int max_fn = 1 << s.log2_max_frame_num;
    c.desc.pp.idr = h.idr();
    bool long_marked = false;
    bool mmco5 = false;
    if (h.idr()) {
      for (const FsPtr& f : dpb)
        if (f.get() != &fs) f->ref_short = f->ref_long = 0;
      if (h.long_term_reference) {
        fs.ref_long |= mask;
        fs.long_idx = 0;
        max_long_idx = 0;
        long_marked = true;
      } else {
        max_long_idx = -1;
      }
    } else if (h.adaptive_marking) {
      const bool field = c.structure != 0;
      const int curr_pic_num = field ? 2 * h.frame_num + 1 : h.frame_num;
      const int cur_par = c.structure == 2 ? 1 : 0;
      auto find_short = [&](int pic_num, FsPtr* out, int* m) {
        for (const FsPtr& f : dpb) {
          if (!field) {
            if (f->ref_short == 3 && wrap_of(*f, h.frame_num, max_fn) == pic_num) {
              *out = f;
              *m = 3;
              return true;
            }
          } else {
            for (int p = 0; p < 2; ++p) {
              if (!((f->ref_short >> p) & 1)) continue;
              const int pn = 2 * wrap_of(*f, h.frame_num, max_fn) + (p == cur_par ? 1 : 0);
              if (pn == pic_num) {
                *out = f;
                *m = 1 << p;
                return true;
              }
            }
          }
        }
        return false;
      };
      auto find_long = [&](int lpn, FsPtr* out, int* m) {
        for (const FsPtr& f : dpb) {
          if (!field) {
            if (f->ref_long == 3 && f->long_idx == lpn) {
              *out = f;
              *m = 3;
              return true;
            }
          } else {
            for (int p = 0; p < 2; ++p) {
              if (!((f->ref_long >> p) & 1)) continue;
              const int pn = 2 * f->long_idx + (p == cur_par ? 1 : 0);
              if (pn == lpn) {
                *out = f;
                *m = 1 << p;
                return true;
              }
            }
          }
        }
        return false;
      };
      for (const Mmco& m : h.mmco) {
        FsPtr f;
        int fm = 0;
        switch (m.op) {
          case 1:
            if (find_short(curr_pic_num - (m.a + 1), &f, &fm)) {
              f->ref_short &= ~fm;
              try_free(f);
            }
            break;
          case 2:
            if (find_long(m.a, &f, &fm)) {
              f->ref_long &= ~fm;
              if (!f->ref_long) f->long_idx = -1;
              try_free(f);
            }
            break;
          case 3:
            if (find_short(curr_pic_num - (m.a + 1), &f, &fm)) {
              for (const FsPtr& g : dpb)
                if (g != f && g->ref_long && g->long_idx == m.b) {
                  g->ref_long = 0;
                  g->long_idx = -1;
                }
              if (f->ref_long && f->long_idx != m.b) {
                f->ref_long = 0;
              }
              f->ref_short &= ~fm;
              f->ref_long |= fm;
              f->long_idx = m.b;
            }
            break;
          case 4:
            max_long_idx = m.a - 1;
            for (const FsPtr& g : dpb)
              if (g->ref_long && g->long_idx > max_long_idx) {
                g->ref_long = 0;
                g->long_idx = -1;
              }
            break;
          case 5:
            for (const FsPtr& g : dpb) {
              if (g.get() == &fs) continue;
              g->ref_short = g->ref_long = 0;
              g->long_idx = -1;
            }
            fs.ref_short = fs.ref_long = 0;
            max_long_idx = -1;
            mmco5 = true;
            break;
          case 6:
            for (const FsPtr& g : dpb)
              if (g.get() != &fs && g->ref_long && g->long_idx == m.b) {
                g->ref_long = 0;
                g->long_idx = -1;
              }
            if (fs.ref_long && fs.long_idx != m.b) fs.ref_long = 0;
            fs.ref_short &= ~mask;
            fs.ref_long |= mask;
            fs.long_idx = m.b;
            long_marked = true;
            break;
          default:
            break;
        }
      }
    } else {
      // sliding window, except for the second field of a pair whose first field is a short-term reference
      const bool second_of_ref = c.second_field && (fs.ref_short != 0);
      if (!second_of_ref) sliding_window(h.frame_num, s, &fs);
    }
    if (!long_marked) fs.ref_short |= mask;
    if (mmco5) {
      // 8.2.1: the picture is treated as having frame_num 0 and its order counts are made relative to itself
      int temp;
      if (c.structure == 0) {
        temp = std::min(fs.poc[0], fs.poc[1]);
        fs.poc[0] -= temp;
        fs.poc[1] -= temp;
      } else {
        const int p = c.structure == 2 ? 1 : 0;
        temp = fs.poc[p];
        fs.poc[p] = 0;
      }
      fs.frame_num = 0;
      c.poc[0] = fs.poc[0];
      c.poc[1] = fs.poc[1];
    }
    prev_had_mmco5 = mmco5;
    for (const FsPtr& f : dpb)
      if (f.get() != &fs) try_free(f);
    // POC state for the next picture
    if (s.poc_type == 0) {
      if (mmco5) {
        prev_poc_msb = 0;
        prev_poc_lsb = c.structure == 2 ? 0 : fs.poc[0];
      } else {
        prev_poc_msb = c.poc_msb;
        prev_poc_lsb = h.poc_lsb;
      }
    }
    prev_ref_frame_num = mmco5 ? 0 : h.frame_num;
  }

  // ---- begin / finish a picture ----
  bool begin_picture(const SliceHeader& h, const std::shared_ptr<Pps>& pps, const std::shared_ptr<Sps>& sps, uint64_t abs_offset) {
    // a pending first field whose second field did not follow: complete its output
    bool second_field = false;
    FsPtr pair_fs;
    if (h.field_pic && pending_first_field) {
      const FrameStore& pf = *pending_first_field;
      const int want_have = h.bottom_field ? 1 : 2;   // the first field must be of the opposite parity
      if (pf.have == want_have && pf.frame_num == h.frame_num && !h.idr()) {
        second_field = true;
        pair_fs = pending_first_field;
      }
    }
    if (!second_field && pending_first_field) {
      pending_first_field.reset();
      bump_after_store();
    }
    const bool first = !have_info;
    const SeqInfo info = make_info(*sps);
    const bool changed = first || !same_info(info, last_info);
    if (changed) {
      // a new sequence: the pictures of the old one are shown first
      if (!first) flush_output(true);
      last_info = info;
      have_info = true;
      const int ret = sink->sequence(info);
      pool = ret > 1 ? ret : std::max(1, info.min_surfaces);
      free_list.clear();
      held.reset();
      for (int i = 0; i < pool; ++i) free_list.push_back(i);
      for (int i = 0; i < 16; ++i) slots[i].reset();
      dpb.clear();
    }
    active_sps = sps;
    // output reordering depth: the stream's own when it says, else the DPB size by level; none for POC type 2
    {
      int dpb_size = dpb_frames_of(*sps);
      if (sps->restriction_present && sps->max_dec_frame_buffering > 0) dpb_size = sps->max_dec_frame_buffering;
      dpb_size = std::max(dpb_size, sps->num_ref_frames);
      dpb_size_ = std::max(1, dpb_size);
      if (sps->restriction_present) reorder_thr = sps->max_dec_frame_buffering > 0 ? std::min(sps->num_reorder_frames, sps->max_dec_frame_buffering) : sps->num_reorder_frames;
      else if (sps->poc_type == 2) reorder_thr = 0;
      else reorder_thr = dpb_size;
    }
    if (h.idr()) {
      // IDR: the references go, then everything before it is output
      for (const FsPtr& f : dpb) {
        f->ref_short = f->ref_long = 0;
        try_free(f);
      }
      while (bump_one()) {
      }
      pop_queue(eff_queue());
      remove_unused();
      prev_frame_num_offset = 0;
      prev_frame_num = 0;
    } else if (!second_field) {
      const int max_fn = 1 << sps->log2_max_frame_num;
      if (h.frame_num != prev_ref_frame_num && h.frame_num != (prev_ref_frame_num + 1) % max_fn && sps->gaps_allowed) fill_frame_num_gap(h, *sps);
    }
    cur = std::make_unique<Cur>();
    Cur& c = *cur;
    c.sps = sps;
    c.pps = pps;
    c.first = c.last = h;
    c.second_field = second_field;
    c.structure = h.field_pic ? (h.bottom_field ? 2 : 1) : 0;
    c.pic_struct = pending_pic_struct;
    pending_pic_struct = -1;
    c.sei = std::move(pending_sei);
    pending_sei.clear();
    compute_poc(c, h, *sps);
    if (second_field) {
      c.fs = pair_fs;
      pending_first_field.reset();
    } else {
      c.fs = std::make_shared<FrameStore>();
      c.fs->current = true;
      c.fs->frame_num = h.frame_num;
      // timestamp of this picture
      int64_t ts;
      bool taken = false;
      ts = 0;
      for (auto it = marks.rbegin(); it != marks.rend(); ++it)
        if (it->at <= abs_offset) {
          if (!it->used) {
            it->used = true;
            ts = it->ts;
            taken = true;
          }
          break;
        }
      if (!taken) {
        if (have_last_ts) ts = last_pic_ts + duration(info);
        else ts = 0;
      }
      while (marks.size() > 1 && marks[1].at <= abs_offset) marks.pop_front();
      last_pic_ts = ts;
      have_last_ts = true;
      ts_heap.push(ts);
    }
    FrameStore& fs = *c.fs;
    if (c.structure == 0) {
      fs.poc[0] = c.poc[0];
      fs.poc[1] = c.poc[1];
    } else {
      fs.poc[c.structure - 1] = c.poc[c.structure - 1];
    }
    // the picture parameters handed to the decode callback
    fill_pic_params(c, h, *sps, *pps);
    return true;
  }

  int64_t duration(const SeqInfo& info) const {
    unsigned num = info.fps_num ? info.fps_num : 30, den = info.fps_num ? info.fps_den : 1;
    return static_cast<int64_t>(clock_rate) * den / num;
  }

  void fill_pic_params(Cur& c, const SliceHeader& h, const Sps& s, const Pps& p) {
    PicParams& pp = c.desc.pp;
    pp.log2_max_frame_num = s.log2_max_frame_num;
    pp.poc_type = s.poc_type;
    pp.log2_max_poc_lsb = s.log2_max_poc_lsb;
    pp.delta_pic_order_always_zero = s.delta_pic_order_always_zero;
    pp.frame_mbs_only = s.frame_mbs_only;
    pp.direct_8x8_inference = s.direct_8x8_inference;
    pp.num_ref_frames = s.num_ref_frames;
    pp.chroma_format_idc = s.chroma_format_idc;
    pp.bit_depth_luma = s.bit_depth_luma;
    pp.bit_depth_chroma = s.bit_depth_chroma;
    pp.transform_bypass = s.qpprime_bypass;
    pp.cabac = p.cabac;
    pp.bottom_field_pic_order_in_frame_present = p.bottom_field_pic_order_in_frame_present;
    pp.redundant_pic_cnt_present = p.redundant_pic_cnt_present;
    pp.num_ref_idx_default[0] = p.num_ref_idx_default[0];
    pp.num_ref_idx_default[1] = p.num_ref_idx_default[1];
    pp.weighted_pred = p.weighted_pred;
    pp.weighted_bipred_idc = p.weighted_bipred_idc;
    pp.pic_init_qp = p.pic_init_qp;
    pp.deblocking_control_present = p.deblocking_control_present;
    pp.constrained_intra_pred = p.constrained_intra_pred;
    pp.transform_8x8_mode = p.transform_8x8_mode;
    pp.chroma_qp_offset[0] = p.chroma_qp_offset[0];
    pp.chroma_qp_offset[1] = p.chroma_qp_offset[1];
    pp.num_slice_groups = p.num_slice_groups;
    pp.slice_group_map_type = p.slice_group_map_type;
    resolve_scaling(s, p, pp.ws4, pp.ws8);
    pp.mbs_w = s.width_mbs;
    pp.mbs_h = s.frame_height_mbs();
    pp.field_pic = h.field_pic;
    pp.bottom_field = h.bottom_field;
    pp.second_field = c.second_field;
    pp.mbaff = s.mbaff && !h.field_pic;
    pp.frame_num = h.frame_num;
    pp.ref_pic = h.nal_ref_idc != 0;
    pp.idr = h.idr();
    c.desc.field_pic = h.field_pic;
    c.desc.bottom_field = h.bottom_field;
    c.desc.second_field = c.second_field;
    c.desc.ref_pic = h.nal_ref_idc != 0;
    c.desc.sps = c.sps.get();
  }

  void finish_picture() {
    if (!cur) return;
    std::unique_ptr<Cur> own = std::move(cur);
    Cur& c = *own;
    if (c.desc.slice_offsets.empty()) {
      // only redundant slices were seen: nothing to decode
      return;
    }
    FrameStore& fs = *c.fs;
    PicDesc& d = c.desc;
    // the picture's surface is taken just before it is decoded
    if (!c.second_field) {
      const int idx = alloc_pic_idx();
      if (idx < 0) return;
      fs.pic_idx = idx;
    }
    d.curr_pic_idx = fs.pic_idx;
    if (want_sei && !c.sei.empty()) sink->sei(fs.pic_idx, c.sei);
    d.intra_pic = c.all_intra;
    if (c.structure == 0) {
      d.pp.poc[0] = c.poc[0];
      d.pp.poc[1] = c.poc[1];
    } else {
      // a field picture: CurrFieldOrderCnt holds the field's own count, and zero for the other parity (measured)
      d.pp.poc[0] = d.pp.poc[1] = 0;
      d.pp.poc[c.structure - 1] = c.second_field ? fs.poc[c.structure - 1] : c.poc[c.structure - 1];
    }
    // the reference pictures as DPB slots
    for (int i = 0; i < 16; ++i) {
      DpbSlot& sl = d.slots[i];
      sl = DpbSlot{};
      const FsPtr& f = slots[i];
      if (!f || !f->ref_any()) continue;
      sl.pic_idx = f->non_existing ? -1 : f->pic_idx;
      sl.frame_idx = f->ref_long ? f->long_idx : f->frame_num;
      sl.is_long_term = f->ref_long != 0;
      sl.not_existing = f->non_existing;
      sl.used = f->ref_any() & 3;
      sl.poc[0] = f->poc[0];
      sl.poc[1] = f->poc[1];
    }
    const bool show = sink->decode(d) != 0;   // an application that returns 0 from the decode callback gets no display callback
    // ---- after decoding: marking, storing, output ----
    fs.have |= c.structure == 0 ? 3 : c.structure;
    const bool first_field = c.structure != 0 && !c.second_field;
    fs.need_output = !fs.non_existing && (show || c.second_field);
    if (c.first.nal_ref_idc != 0) {
      mark_current(c);
    } else {
      prev_had_mmco5 = false;
    }
    if (c.sps->poc_type != 0) {
      prev_frame_num_offset = prev_had_mmco5 ? 0 : c.frame_num_offset;
      prev_frame_num = prev_had_mmco5 ? 0 : c.first.frame_num;
    }
    // display parameters of the frame
    {
      const Sps& s = *c.sps;
      bool prog = s.frame_mbs_only != 0;
      bool tff = false;
      int repeat = 0;
      if (!s.frame_mbs_only) {
        prog = false;
        if (c.pic_struct >= 0) {
          prog = c.pic_struct == 0;
          tff = c.pic_struct == 3 || c.pic_struct == 5;
          repeat = (c.pic_struct == 5 || c.pic_struct == 6 || c.pic_struct == 7) ? 1 : (c.pic_struct == 8 ? 2 : 0);
        } else if (c.structure == 0) {
          prog = true;
        } else {
          tff = c.structure == 1;   // the first field of the pair
        }
      } else if (c.pic_struct >= 0) {
        repeat = (c.pic_struct == 7) ? 1 : (c.pic_struct == 8 ? 2 : 0);
      }
      if (!c.second_field) {
        fs.progressive = prog;
        fs.tff = tff;
        fs.repeat = repeat;
      }
    }
    remove_unused();
    fs.current = false;
    if (first_field) {
      store_current(c.fs);
      pending_first_field = c.fs;
    } else {
      pending_first_field.reset();
      store_and_bump(c.fs);
      try_free(c.fs);
    }
  }

  // The picture goes into the DPB (a reference, or waiting for output) -- after the pictures that must leave to make room
  // have been output, so a picture takes the place of one that just went.
  void store_current(const FsPtr& f) {
    if (f->in_dpb) return;
    f->in_dpb = true;
    dpb.push_back(f);
    assign_slot(f);
  }

  void store_and_bump(const FsPtr& f) {
    // 1. room for the picture (C.4.5.3): while the DPB is full, the picture next in output order leaves
    if (!f->in_dpb && (f->need_output || f->ref_any())) {
      while (static_cast<int>(dpb.size()) >= dpb_size_) {
        FsPtr best;
        for (const FsPtr& g : dpb)
          if (g->need_output && (!best || frame_poc(*g) < frame_poc(*best))) best = g;
        if (!best) break;
        if (f->need_output && !f->ref_any() && frame_poc(*f) < frame_poc(*best)) {
          // a non-reference picture that is next in output order is output without being stored (C.4.5.2)
          f->need_output = false;
          f->queued = true;
          out_queue.push_back(f);
          break;
        }
        bump_one();
      }
    }
    // 2. store it
    if (!f->in_dpb && (f->need_output || f->ref_any())) store_current(f);
    // 3. the stream's reordering depth
    while (waiting_count() > eff_thr() && bump_one()) {
    }
    pop_queue(eff_queue());
    remove_unused();
  }

  // The display delay the card applies: the application's value up to 3 (NVDEC's decode queue depth).
  size_t eff_queue() const { return std::min<size_t>(max_delay, 3); }
  int eff_thr() const { return reorder_thr; }

  void bump_after_store() {
    remove_unused();
    while (waiting_count() > eff_thr() && bump_one()) {
    }
    pop_queue(eff_queue());
    remove_unused();
  }
};

H264Parser::H264Parser(ParserSink* sink, unsigned max_decode_surfaces, unsigned clock_rate, unsigned max_display_delay, bool want_sei)
    : p_(new Impl(sink, max_decode_surfaces, clock_rate, max_display_delay, want_sei)) {}
H264Parser::~H264Parser() = default;
void H264Parser::feed(const uint8_t* data, size_t n, bool has_ts, int64_t ts, bool discontinuity, bool end_of_picture) {
  p_->feed(data, n, has_ts, ts, discontinuity, end_of_picture);
}
void H264Parser::end_of_stream() { p_->end_of_stream(); }

}  // namespace vgpu_h264
