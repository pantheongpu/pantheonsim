// See hevc_parser.hpp.
#include "hevc_parser.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <functional>

namespace vgpu_hevc {

namespace {

struct Pic {
  int pic_idx = -1;
  int poc = 0;
  bool ref_short = false, ref_long = false;
  bool need_output = false;
  int latency = 0;
  int slot = -1;
  bool in_dpb = false;
  bool queued = false;         // bumped, waiting in the display queue
  bool never_output = false;   // PicOutputFlag is 0: the picture is not displayed
  bool displayed = false;      // its display callback has been made (or it is not output at all)
  int disp_seq = 0;            // the display count at that moment
  bool current = false;        // the picture being decoded or stored
  bool progressive = true, tff = false;
  int repeat = 0;
  bool ref_any() const { return ref_short || ref_long; }
};
using PicPtr = std::shared_ptr<Pic>;

int gcd_int(int a, int b) {
  while (b) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

struct ParamRaw {
  std::vector<uint8_t> raw;
};

}  // namespace

struct HevcParser::Impl {
  ParserSink* sink;
  unsigned max_surfaces, clock_rate, max_delay;
  bool want_sei;

  // ---- input ----
  std::vector<uint8_t> buf;
  uint64_t buf_base = 0;
  uint64_t total_fed = 0;
  struct TsMark {
    uint64_t at;
    int64_t ts;
    bool used;
  };
  std::deque<TsMark> marks;
  bool peeked_this_nal = false;

  // ---- parameter sets ----
  ParamRaw sps_raw[16], pps_raw[64];
  std::shared_ptr<Sps> sps_[16];
  std::shared_ptr<Pps> pps_[64];

  // ---- sequence ----
  std::shared_ptr<Sps> active_sps;
  SeqInfo last_info;
  bool have_info = false;
  int pool = 0;
  std::deque<int> pool_order;   // the pool's surfaces in the order they are tried
  std::vector<PicPtr> surf_pic; // the picture each surface holds
  int display_count = 0, hold_displays = 3;
  PicPtr last_pic;              // the picture decoded last

  // ---- DPB ----
  std::vector<PicPtr> dpb;
  PicPtr slots[16];
  std::deque<PicPtr> out_queue;
  std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>> ts_heap;
  int64_t last_pic_ts = 0;
  bool have_last_ts = false;

  // ---- decoding order state ----
  int prev_poc_lsb = 0, prev_poc_msb = 0;
  bool first_pic = true, after_eos = false, assoc_irap_no_rasl = false;

  // ---- the picture being assembled ----
  struct Cur {
    std::shared_ptr<Sps> sps;
    std::shared_ptr<Pps> pps;
    SliceHeader first, last;
    int nal_type = 0, tid = 0;
    bool skip = false;
    bool no_output = false;      // PicOutputFlag forced to 0
    bool no_rasl_output = false;
    PicDesc desc;
    bool all_intra = true;
    int64_t ts = 0;
    std::vector<SeiMessage> sei;
    // filled by prepare_picture(): the picture order count and the reference picture set
    bool prepared = false;
    int poc = 0;
    std::vector<PicPtr> st_before, st_after, lt_curr;
  };
  std::unique_ptr<Cur> cur;
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
        if (buf.size() > 2) {
          buf_base += buf.size() - 2;
          buf.erase(buf.begin(), buf.end() - 2);
        }
        return;
      }
      int sc_first = s1;
      if (sc_first > 0 && buf[sc_first - 1] == 0) --sc_first;
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
      size_t end = buf.size();
      if (eos) {
        while (end > static_cast<size_t>(s1) + 3 && buf[end - 1] == 0) --end;
        if (end > static_cast<size_t>(s1) + 3) consume_nal(buf.data() + s1 + 3, end - (s1 + 3), end - (s1 + 3), buf_base + static_cast<uint64_t>(sc_first));
        buf_base += buf.size();
        buf.clear();
        return;
      }
      // NVIDIA's parser acts on a NAL unit before it is complete: a slice segment, a VPS or an SPS once 13 bytes of it are buffered
      // (the slice's first_slice_segment_in_pic_flag says whether it starts a picture; the others start an access unit), and
      // a prefix SEI message once 56 bytes of it are (measured with the NAL units cut at every byte; shorter SEI messages
      // and the other non-VCL NAL units act when the next start code ends them, and suffix SEI messages not at all).
      if (!peeked_this_nal) {
        const size_t have = end - static_cast<size_t>(s1) - 3;
        if (have >= 2) {
          const uint8_t* nal = buf.data() + s1 + 3;
          const int type = (nal[0] >> 1) & 63;
          if (have >= 13 && (type <= 9 || (type >= 16 && type <= 21))) {
            peeked_this_nal = true;
            peek_slice(nal, have);
          } else if (have >= 13 && (type == kVps || type == kSps)) {
            peeked_this_nal = true;
            finish_picture();
          } else if (have >= 56 && type == kPrefixSei) {
            peeked_this_nal = true;
            finish_picture();
          }
        }
      }
      return;
    }
  }

  void peek_slice(const uint8_t* nal, size_t n) {
    if (!cur || n < 3) return;
    const int type = (nal[0] >> 1) & 63;
    const int layer = ((nal[0] & 1) << 5) | (nal[1] >> 3);
    if (layer != 0) return;
    std::vector<uint8_t> rb = unescape(nal, std::min<size_t>(n, 16), 2);
    int pps_id;
    bool first;
    if (!peek_slice_pps_id(rb.data(), rb.size(), type, &pps_id, &first)) return;
    if (first) finish_picture();
  }

  // ================================================================== NAL units
  void consume_nal(const uint8_t* nal, size_t n, size_t n_raw, uint64_t abs_offset) {
    if (n < 2) return;
    const int type = (nal[0] >> 1) & 63;
    const int layer = ((nal[0] & 1) << 5) | (nal[1] >> 3);
    if (layer != 0) return;   // only the base layer is decoded
    if (type < 32) {
      if (type <= 9 || (type >= 16 && type <= 21)) consume_slice(nal, n, n_raw, abs_offset);
      return;
    }
    switch (type) {
      case kSps: {
        finish_picture();
        std::vector<uint8_t> rb = unescape(nal, n, 2);
        auto s = std::make_shared<Sps>();
        if (!parse_sps(rb.data(), rb.size(), s.get())) break;
        const int id = s->id;
        if (sps_raw[id].raw != rb || !sps_[id]) {
          sps_raw[id].raw = rb;
          sps_[id] = s;
        }
        break;
      }
      case kVps:
        finish_picture();
        break;
      case kPps: {
        finish_picture();
        std::vector<uint8_t> rb = unescape(nal, n, 2);
        auto p = std::make_shared<Pps>();
        if (!parse_pps(rb.data(), rb.size(), p.get())) break;
        pps_raw[p->id].raw = rb;
        pps_[p->id] = p;
        break;
      }
      case kAud:
        finish_picture();
        break;
      case kEos:
      case kEob:
        finish_picture();   // measured: the card does nothing more -- the pictures before it stay in the DPB, a CRA picture after it is no random access point
        break;
      case kPrefixSei:
      case kSuffixSei:
        parse_sei(nal, n);
        break;
      default:
        break;
    }
  }

  void parse_sei(const uint8_t* nal, size_t n) {
    std::vector<uint8_t> rb = unescape(nal, n, 2);
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
      pending_sei.push_back(std::move(m));
      i += static_cast<size_t>(size);
    }
  }

  // ================================================================== slices and pictures
  static SliceCtx make_ctx(const Sps& s, const Pps& p) {
    SliceCtx c;
    c.log2_ctb = s.log2_ctb;
    c.width_ctbs = s.width_ctbs();
    c.height_ctbs = s.height_ctbs();
    c.log2_max_poc_lsb = s.log2_max_poc_lsb;
    c.num_st_rps = static_cast<int>(s.st_rps.size());
    c.sps_rps = &s.st_rps;
    c.long_term_present = s.long_term_present;
    c.num_lt_sps = s.num_lt_sps;
    c.lt_poc_lsb_sps = s.lt_poc_lsb_sps;
    c.lt_used_sps = s.lt_used_sps;
    c.temporal_mvp = s.temporal_mvp;
    c.sao = s.sao;
    c.chroma_array_type = s.chroma_array_type();
    c.separate_colour_plane = s.separate_colour_plane != 0;
    c.high_precision_offsets = s.high_precision_offsets;
    c.bit_depth_chroma = s.bit_depth_chroma;
    c.dependent_slice_segments_enabled = p.dependent_slice_segments_enabled;
    c.output_flag_present = p.output_flag_present;
    c.num_extra_slice_header_bits = p.num_extra_slice_header_bits;
    c.lists_modification_present = p.lists_modification_present;
    c.cabac_init_present = p.cabac_init_present;
    c.num_ref_idx_default[0] = p.num_ref_idx_default[0];
    c.num_ref_idx_default[1] = p.num_ref_idx_default[1];
    c.weighted_pred = p.weighted_pred;
    c.weighted_bipred = p.weighted_bipred;
    c.slice_chroma_qp_offsets_present = p.slice_chroma_qp_offsets_present;
    c.chroma_qp_offset_list_enabled = p.chroma_qp_offset_list_enabled;
    c.deblocking_override_enabled = p.deblocking_override_enabled;
    c.pps_deblocking_disabled = p.deblocking_disabled;
    c.pps_beta_offset_div2 = p.beta_offset_div2;
    c.pps_tc_offset_div2 = p.tc_offset_div2;
    c.pps_loop_filter_across_slices = p.loop_filter_across_slices;
    c.tiles_enabled = p.tiles_enabled;
    c.entropy_coding_sync = p.entropy_coding_sync;
    c.slice_header_extension_present = p.slice_header_extension_present;
    c.init_qp = p.init_qp;
    return c;
  }

  void consume_slice(const uint8_t* nal, size_t n, size_t n_raw, uint64_t abs_offset) {
    if (n < 3) return;
    const int type = (nal[0] >> 1) & 63;
    const int tid = (nal[1] & 7) - 1;
    if (tid < 0) return;
    std::vector<uint8_t> rb = unescape(nal, n, 2);
    int pps_id;
    bool first_flag;
    if (!peek_slice_pps_id(rb.data(), rb.size(), type, &pps_id, &first_flag)) return;
    if (first_flag) finish_picture();
    if (!pps_[pps_id]) return;
    std::shared_ptr<Pps> pps = pps_[pps_id];
    std::shared_ptr<Sps> sps = sps_[pps->sps_id];
    if (!sps) return;
    if (first_flag) {
      if (!begin_picture(type, tid, sps, pps, abs_offset)) return;
    }
    if (!cur) return;
    if (cur->skip) return;
    // the header, for the picture's first slice and for the slice type of every slice
    BitReader br(rb.data(), rb.size());
    SliceHeader h;
    const SliceCtx ctx = make_ctx(*cur->sps, *cur->pps);
    if (!parse_slice_header(br, ctx, type, cur->desc.slice_nal.empty() ? nullptr : &cur->last, &h)) return;
    if (first_flag) {
      cur->first = h;
      if (h.pic_output_flag && !cur->no_output) ts_heap.push(cur->ts);
      prepare_picture(*cur);
    }
    if (!h.dependent) cur->last = h;
    if (!h.is_intra()) cur->all_intra = false;
    PicDesc& d = cur->desc;
    d.slice_offsets.push_back(static_cast<unsigned>(d.data.size()));
    d.data.insert(d.data.end(), {0, 0, 1});
    d.slice_nal.push_back({d.data.size(), n});
    d.data.insert(d.data.end(), nal, nal + n_raw);
  }

  // The sequence information for an SPS.
  SeqInfo make_info(const Sps& s) {
    SeqInfo i;
    i.coded_w = (s.width + 15) & ~15;   // measured: the sequence's coded size is the picture size rounded up to whole 16x16 blocks
    i.coded_h = (s.height + 15) & ~15;
    const int cux = s.sub_width_c(), cuy = s.sub_height_c();
    i.disp_left = s.conf_win[0] * cux;
    i.disp_right = s.width - s.conf_win[1] * cux;
    i.disp_top = s.conf_win[2] * cuy;
    i.disp_bottom = s.height - s.conf_win[3] * cuy;
    if (!(i.disp_right > i.disp_left && i.disp_bottom > i.disp_top && i.disp_left >= 0 && i.disp_top >= 0)) {
      i.disp_left = i.disp_top = 0;
      i.disp_right = s.width;
      i.disp_bottom = s.height;
    }
    i.progressive = !s.field_seq;
    i.chroma_format = s.chroma_format_idc;
    i.bit_depth_luma_minus8 = s.bit_depth_luma - 8;
    i.bit_depth_chroma_minus8 = s.bit_depth_chroma - 8;
    i.profile = s.profile_idc;
    i.level = s.level_idc;
    i.min_surfaces = min_surfaces_of(s);
    if (s.timing_present && s.num_units_in_tick && s.time_scale) {
      const unsigned g = static_cast<unsigned>(gcd_int(static_cast<int>(s.time_scale), static_cast<int>(s.num_units_in_tick)));
      i.fps_num = s.time_scale / g;
      i.fps_den = s.num_units_in_tick / g;
    }
    i.bitrate = s.hrd_bit_rate;
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
    i.supported = s.chroma_format_idc == 1 && !s.separate_colour_plane && s.bit_depth_luma <= 12 && s.bit_depth_chroma <= 12 && !s.unsupported_extension;
    return i;
  }

  static int min_surfaces_of(const Sps& s) {
    const int hi = s.highest_tid();
    return std::max(1, s.max_dec_pic_buffering[hi] + 4);
  }

  static bool same_info(const SeqInfo& a, const SeqInfo& b) {
    return a.coded_w == b.coded_w && a.coded_h == b.coded_h && a.disp_left == b.disp_left && a.disp_top == b.disp_top && a.disp_right == b.disp_right &&
           a.disp_bottom == b.disp_bottom && a.fps_num == b.fps_num && a.fps_den == b.fps_den && a.progressive == b.progressive &&
           a.chroma_format == b.chroma_format && a.bit_depth_luma_minus8 == b.bit_depth_luma_minus8 && a.bit_depth_chroma_minus8 == b.bit_depth_chroma_minus8 &&
           a.min_surfaces == b.min_surfaces && a.bitrate == b.bitrate && a.dar_x == b.dar_x && a.dar_y == b.dar_y && a.video_format == b.video_format &&
           a.full_range == b.full_range && a.primaries == b.primaries && a.transfer == b.transfer && a.matrix == b.matrix;
  }

  // ---- surfaces ----
  // A surface is given out again when the picture it holds is no longer a reference, has been displayed, and was displayed at least
  // `hold_displays` displays ago (max_num_reorder + 3, measured); surfaces are tried in the order they were first given out.
  // When none qualifies, the picture displayed longest ago goes.
  bool surface_free(const PicPtr& f, int age) const {
    if (!f) return true;
    if (f->ref_any() || f->need_output || f->queued || f->current || !f->displayed) return false;
    return display_count - f->disp_seq >= age;
  }

  int take_surface(int s) {
    for (auto it = pool_order.begin(); it != pool_order.end(); ++it)
      if (*it == s) {
        pool_order.erase(it);
        break;
      }
    pool_order.push_back(s);
    surf_pic[static_cast<size_t>(s)].reset();
    return s;
  }

  int alloc_pic_idx() {
    for (;;) {
      // first the surfaces of pictures that were never output (they go as soon as they are no longer references, and the card reuses
      // those rather than a fresh surface), then the others in the order they were first given out: the ones never used come
      // first, then the surfaces used before, oldest first
      for (int pass = 0; pass < 2; ++pass) {
        int best = -1;
        for (int s : pool_order) {
          const PicPtr& f = surf_pic[static_cast<size_t>(s)];
          if (pass == 0 && !(f && f->never_output)) continue;
          if (!surface_free(f, hold_displays)) continue;
          if (!f) return take_surface(s);
          if (best < 0 || f->disp_seq < surf_pic[static_cast<size_t>(best)]->disp_seq) best = s;
        }
        if (best >= 0) return take_surface(best);
      }
      int best = -1;
      for (int s = 0; s < pool; ++s) {
        const PicPtr& f = surf_pic[static_cast<size_t>(s)];
        if (f && surface_free(f, 0) && (best < 0 || f->disp_seq < surf_pic[static_cast<size_t>(best)]->disp_seq)) best = s;
      }
      if (best >= 0) return take_surface(best);
      if (!out_queue.empty()) {
        display_one();
      } else if (bump_one()) {
        display_one();
      } else {
        return -1;
      }
    }
  }

  // ---- output (C.5.2.4) ----
  int waiting_count() const {
    int c = 0;
    for (const PicPtr& f : dpb)
      if (f->need_output) ++c;
    return c;
  }

  bool bump_one() {
    PicPtr best;
    for (const PicPtr& f : dpb)
      if (f->need_output && (!best || f->poc < best->poc)) best = f;
    if (!best) return false;
    best->need_output = false;
    best->queued = true;
    out_queue.push_back(best);
    remove_unused();
    return true;
  }

  void display_one() {
    PicPtr f = out_queue.front();
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
    f->displayed = true;
    f->disp_seq = ++display_count;
  }

  void pop_queue(size_t limit) {
    while (out_queue.size() > limit) display_one();
  }

  void remove_unused() {
    for (size_t i = 0; i < dpb.size();) {
      const PicPtr& f = dpb[i];
      if (!f->ref_any() && !f->need_output && !f->current) {
        if (f->slot >= 0 && slots[f->slot] == f) slots[f->slot].reset();
        f->slot = -1;
        f->in_dpb = false;
        dpb.erase(dpb.begin() + static_cast<long>(i));
      } else {
        ++i;
      }
    }
  }

  void assign_slot(const PicPtr& f) {
    if (f->slot >= 0) return;
    for (int i = 0; i < 16; ++i)
      if (!slots[i]) {
        slots[i] = f;
        f->slot = i;
        return;
      }
  }

  void flush_output(bool all) {
    while (bump_one()) {
    }
    if (all) pop_queue(0);
  }

  size_t eff_queue() const { return std::min<size_t>(max_delay, 3); }

  bool need_bump(const Sps& s) const {
    const int hi = s.highest_tid();
    if (waiting_count() > s.max_num_reorder[hi]) return true;
    if (s.max_latency_increase_plus1[hi]) {
      const int max_latency = s.max_num_reorder[hi] + s.max_latency_increase_plus1[hi] - 1;
      for (const PicPtr& f : dpb)
        if (f->need_output && f->latency >= max_latency) return true;
    }
    return false;
  }

  // ---- begin / finish a picture ----
  bool begin_picture(int type, int tid, const std::shared_ptr<Sps>& sps, const std::shared_ptr<Pps>& pps, uint64_t abs_offset) {
    cur = std::make_unique<Cur>();
    Cur& c = *cur;
    c.sps = sps;
    c.pps = pps;
    c.nal_type = type;
    c.tid = tid;
    if (is_irap(type)) {
      c.no_rasl_output = is_idr(type) || is_bla(type) || first_pic || after_eos;
      assoc_irap_no_rasl = c.no_rasl_output;
    }
    // RASL pictures of a random access point decoding starts at are decoded (the card's parser hands them to the decoder with the
    // references it has) but not output (PicOutputFlag is 0, 8.1.3)
    c.no_output = is_rasl(type) && assoc_irap_no_rasl;
    // ---- sequence change: announced when the first slice segment of the picture is read
    {
      const Sps& s = *c.sps;
      const int hi = s.highest_tid();
      const SeqInfo info = make_info(s);
      const bool first = !have_info;
      const bool changed = first || !same_info(info, last_info);
      if (changed) {
        if (!first) flush_output(true);
        last_info = info;
        have_info = true;
        const int ret = sink->sequence(info);
        pool = ret > 1 ? ret : std::max(1, info.min_surfaces);
        // the pictures of the old sequence are all output; the surfaces keep their state (the card's first picture of a new sequence
        // takes a surface that was never used, and then the one displayed longest ago, as before)
        for (const PicPtr& f : dpb) f->ref_short = f->ref_long = false;
        for (int i = 0; i < 16; ++i) slots[i].reset();
        dpb.clear();
        out_queue.clear();
        std::deque<int> order;
        for (int i : pool_order)
          if (i < pool) order.push_back(i);
        for (int i = static_cast<int>(surf_pic.size()); i < pool; ++i) order.push_back(i);
        pool_order = order;
        surf_pic.resize(static_cast<size_t>(pool));
      }
      active_sps = c.sps;
      hold_displays = s.max_num_reorder[hi] + 3;
    }
    c.sei = std::move(pending_sei);
    pending_sei.clear();
    // timestamp of this picture
    int64_t ts = 0;
    bool taken = false;
    for (auto it = marks.rbegin(); it != marks.rend(); ++it)
      if (it->at <= abs_offset) {
        if (!it->used) {
          it->used = true;
          ts = it->ts;
          taken = true;
        }
        break;
      }
    const SeqInfo info = make_info(*sps);
    if (!taken) {
      if (have_last_ts) {
        const unsigned num = info.fps_num ? info.fps_num : 30, den = info.fps_num ? info.fps_den : 1;
        ts = last_pic_ts + static_cast<int64_t>(clock_rate) * den / num;
      } else {
        ts = 0;
      }
    }
    while (marks.size() > 1 && marks[1].at <= abs_offset) marks.pop_front();
    last_pic_ts = ts;
    have_last_ts = true;
    c.ts = ts;
    return true;
  }

  static void resolve_scaling(const Sps& s, const Pps& p, PicParams* pp) {
    pp->scaling_list_enabled = s.scaling_list_enabled;
    ScalingLists def;
    const ScalingLists* sl = &def;
    if (s.scaling_list_enabled) {
      if (p.scaling_list_data_present) sl = &p.scaling;
      else if (s.scaling_list_data_present) sl = &s.scaling;
    }
    for (int m = 0; m < 6; ++m) {
      if (!s.scaling_list_enabled) {
        std::memset(pp->sl4[m], 16, 16);
        std::memset(pp->sl8[m], 16, 64);
        std::memset(pp->sl16[m], 16, 64);
        pp->dc16[m] = 16;
        continue;
      }
      std::memcpy(pp->sl4[m], sl->list[0][m], 16);
      std::memcpy(pp->sl8[m], sl->list[1][m], 64);
      std::memcpy(pp->sl16[m], sl->list[2][m], 64);
      pp->dc16[m] = static_cast<uint8_t>(sl->dc[0][m]);
    }
    for (int k = 0; k < 2; ++k) {
      if (!s.scaling_list_enabled) {
        std::memset(pp->sl32[k], 16, 64);
        pp->dc32[k] = 16;
        continue;
      }
      std::memcpy(pp->sl32[k], sl->list[3][k * 3], 64);
      pp->dc32[k] = static_cast<uint8_t>(sl->dc[1][k * 3]);
    }
  }

  void fill_pic_params(Cur& c, const SliceHeader& h) {
    const Sps& s = *c.sps;
    const Pps& p = *c.pps;
    PicParams& pp = c.desc.pp;
    pp.width = s.width;
    pp.height = s.height;
    pp.log2_min_cb = s.log2_min_cb;
    pp.log2_diff_cb = s.log2_ctb - s.log2_min_cb;
    pp.log2_min_tb = s.log2_min_tb;
    pp.log2_diff_tb = s.log2_max_tb - s.log2_min_tb;
    pp.max_th_depth_intra = s.max_th_depth_intra;
    pp.max_th_depth_inter = s.max_th_depth_inter;
    pp.amp = s.amp;
    pp.sao = s.sao;
    pp.strong_intra_smoothing = s.strong_intra_smoothing;
    pp.sps_temporal_mvp = s.temporal_mvp;
    pp.pcm = s.pcm;
    pp.pcm_loop_filter_disabled = s.pcm_loop_filter_disabled;
    pp.pcm_bit_depth_luma = s.pcm_bit_depth_luma;
    pp.pcm_bit_depth_chroma = s.pcm_bit_depth_chroma;
    pp.log2_min_pcm = s.log2_min_pcm_cb;
    pp.log2_diff_pcm = s.log2_max_pcm_cb - s.log2_min_pcm_cb;
    pp.bit_depth_luma = s.bit_depth_luma;
    pp.bit_depth_chroma = s.bit_depth_chroma;
    pp.chroma_format_idc = s.chroma_format_idc;
    pp.separate_colour_plane = s.separate_colour_plane != 0;
    pp.log2_max_poc_lsb = s.log2_max_poc_lsb;
    pp.num_short_term_ref_pic_sets = static_cast<int>(s.st_rps.size());
    pp.long_term_ref_pics_present = s.long_term_present;
    pp.num_long_term_ref_pics_sps = s.num_lt_sps;
    pp.irap = is_irap(c.nal_type);
    pp.idr = is_idr(c.nal_type);
    pp.sps_range_extension = s.range_extension;
    pp.pps_range_extension = p.range_extension;
    pp.transform_skip_rotation = s.transform_skip_rotation;
    pp.transform_skip_context = s.transform_skip_context;
    pp.implicit_rdpcm = s.implicit_rdpcm;
    pp.explicit_rdpcm = s.explicit_rdpcm;
    pp.extended_precision = s.extended_precision;
    pp.intra_smoothing_disabled = s.intra_smoothing_disabled;
    pp.high_precision_offsets = s.high_precision_offsets;
    pp.persistent_rice_adaptation = s.persistent_rice_adaptation;
    pp.cabac_bypass_alignment = s.cabac_bypass_alignment;
    pp.log2_max_transform_skip = p.log2_max_transform_skip_size;
    pp.log2_sao_offset_scale_luma = p.log2_sao_offset_scale_luma;
    pp.log2_sao_offset_scale_chroma = p.log2_sao_offset_scale_chroma;
    pp.cross_component_prediction = p.cross_component_prediction;
    pp.chroma_qp_offset_list_enabled = p.chroma_qp_offset_list_enabled;
    pp.diff_cu_chroma_qp_offset_depth = p.diff_cu_chroma_qp_offset_depth;
    pp.chroma_qp_offset_list_len = p.chroma_qp_offset_list_len;
    for (int i = 0; i < 6; ++i) {
      pp.cb_qp_offset_list[i] = p.cb_qp_offset_list[i];
      pp.cr_qp_offset_list[i] = p.cr_qp_offset_list[i];
    }
    pp.dependent_slice_segments_enabled = p.dependent_slice_segments_enabled;
    pp.slice_segment_header_extension_present = p.slice_header_extension_present;
    pp.sign_data_hiding = p.sign_data_hiding;
    pp.cu_qp_delta_enabled = p.cu_qp_delta_enabled;
    pp.diff_cu_qp_delta_depth = p.diff_cu_qp_delta_depth;
    pp.init_qp = p.init_qp;
    pp.cb_qp_offset = p.cb_qp_offset;
    pp.cr_qp_offset = p.cr_qp_offset;
    pp.constrained_intra_pred = p.constrained_intra_pred;
    pp.weighted_pred = p.weighted_pred;
    pp.weighted_bipred = p.weighted_bipred;
    pp.transform_skip_enabled = p.transform_skip_enabled;
    pp.transquant_bypass_enabled = p.transquant_bypass_enabled;
    pp.entropy_coding_sync = p.entropy_coding_sync;
    pp.log2_parallel_merge_level = p.log2_parallel_merge_level;
    pp.num_extra_slice_header_bits = p.num_extra_slice_header_bits;
    pp.loop_filter_across_tiles = p.loop_filter_across_tiles;
    pp.loop_filter_across_slices = p.loop_filter_across_slices;
    pp.output_flag_present = p.output_flag_present;
    pp.num_ref_idx_default[0] = p.num_ref_idx_default[0];
    pp.num_ref_idx_default[1] = p.num_ref_idx_default[1];
    pp.lists_modification_present = p.lists_modification_present;
    pp.cabac_init_present = p.cabac_init_present;
    pp.slice_chroma_qp_offsets_present = p.slice_chroma_qp_offsets_present;
    pp.deblocking_override_enabled = p.deblocking_override_enabled;
    pp.pps_deblocking_disabled = p.deblocking_disabled;
    pp.pps_beta_offset_div2 = p.beta_offset_div2;
    pp.pps_tc_offset_div2 = p.tc_offset_div2;
    pp.tiles_enabled = p.tiles_enabled;
    pp.uniform_spacing = p.uniform_spacing;
    pp.num_tile_cols = p.num_tile_cols;
    pp.num_tile_rows = p.num_tile_rows;
    for (int i = 0; i < 20; ++i) {
      pp.column_width_minus1[i] = p.column_width_minus1[i];
      pp.row_height_minus1[i] = p.row_height_minus1[i];
    }
    pp.num_bits_st_rps = h.st_rps_bits;
    resolve_scaling(s, p, &pp);
  }

  // Everything up to the removal and output before decoding (C.5.2.2): the card does this as soon as the first slice segment of the
  // picture has been consumed, which can be well before the picture is complete (measured with chunked feeding).
  void prepare_picture(Cur& c) {
    c.prepared = true;
    const Sps& s = *c.sps;
    const SliceHeader& h = c.first;
    const int hi = s.highest_tid();
    const bool irap = is_irap(c.nal_type);
    // ---- picture order count (8.3.1)
    const int max_lsb = 1 << s.log2_max_poc_lsb;
    int poc_msb;
    if (irap && c.no_rasl_output) {
      poc_msb = 0;
    } else if (h.poc_lsb < prev_poc_lsb && prev_poc_lsb - h.poc_lsb >= max_lsb / 2) {
      poc_msb = prev_poc_msb + max_lsb;
    } else if (h.poc_lsb > prev_poc_lsb && h.poc_lsb - prev_poc_lsb > max_lsb / 2) {
      poc_msb = prev_poc_msb - max_lsb;
    } else {
      poc_msb = prev_poc_msb;
    }
    const int poc = poc_msb + h.poc_lsb;
    if (c.tid == 0 && !is_rasl(c.nal_type) && !is_radl(c.nal_type) && !is_sub_layer_non_ref(c.nal_type)) {
      prev_poc_lsb = h.poc_lsb;
      prev_poc_msb = poc_msb;
    }
    // ---- reference picture set (8.3.2)
    if (irap && c.no_rasl_output)
      for (const PicPtr& f : dpb) f->ref_short = f->ref_long = false;
    std::vector<PicPtr> st_before, st_after, st_foll, lt_curr, lt_foll;
    std::vector<int> st_before_poc, st_after_poc;
    if (!is_idr(c.nal_type)) {
      std::vector<int> st_foll_poc, lt_curr_poc, lt_foll_poc;
      std::vector<bool> lt_curr_msb, lt_foll_msb;
      for (int i = 0; i < h.st_rps.num_neg; ++i) {
        const int p = poc + h.st_rps.delta_s0[i];
        if (h.st_rps.used_s0[i]) st_before_poc.push_back(p);
        else st_foll_poc.push_back(p);
      }
      for (int i = 0; i < h.st_rps.num_pos; ++i) {
        const int p = poc + h.st_rps.delta_s1[i];
        if (h.st_rps.used_s1[i]) st_after_poc.push_back(p);
        else st_foll_poc.push_back(p);
      }
      for (int i = 0; i < h.num_lt; ++i) {
        int p = h.lt_poc_lsb[i];
        if (h.lt_msb_present[i]) p += poc - h.lt_delta_msb_cycle[i] * max_lsb - (poc & (max_lsb - 1));
        if (h.lt_used[i]) {
          lt_curr_poc.push_back(p);
          lt_curr_msb.push_back(h.lt_msb_present[i]);
        } else {
          lt_foll_poc.push_back(p);
          lt_foll_msb.push_back(h.lt_msb_present[i]);
        }
      }
      auto find_lt = [&](int p, bool msb) -> PicPtr {
        for (const PicPtr& f : dpb)
          if (f->ref_any() && (msb ? f->poc == p : (f->poc & (max_lsb - 1)) == p)) return f;
        return nullptr;
      };
      std::vector<PicPtr> marked_lt;
      for (size_t i = 0; i < lt_curr_poc.size(); ++i) lt_curr.push_back(find_lt(lt_curr_poc[i], lt_curr_msb[i]));
      for (size_t i = 0; i < lt_foll_poc.size(); ++i) lt_foll.push_back(find_lt(lt_foll_poc[i], lt_foll_msb[i]));
      for (const PicPtr& f : lt_curr)
        if (f) f->ref_long = true, f->ref_short = false;
      for (const PicPtr& f : lt_foll)
        if (f) f->ref_long = true, f->ref_short = false;
      auto find_st = [&](int p) -> PicPtr {
        for (const PicPtr& f : dpb)
          if (f->ref_short && f->poc == p) return f;
        return nullptr;
      };
      for (int p : st_before_poc) st_before.push_back(find_st(p));
      for (int p : st_after_poc) st_after.push_back(find_st(p));
      for (int p : st_foll_poc) st_foll.push_back(find_st(p));
      // a reference picture that is not there (a RASL picture after a random access point, a damaged stream) is replaced by the picture
      // decoded last (measured on the card's picture tables)
      if (last_pic)
        for (auto* v : {&st_before, &st_after, &lt_curr})
          for (PicPtr& f : *v)
            if (!f) f = last_pic;
      // everything else is no longer a reference
      for (const PicPtr& f : dpb) {
        bool keep = false;
        for (const auto* v : {&st_before, &st_after, &st_foll, &lt_curr, &lt_foll})
          for (const PicPtr& g : *v)
            if (g == f) keep = true;
        if (!keep) f->ref_short = f->ref_long = false;
      }
    } else {
      for (const PicPtr& f : dpb) f->ref_short = f->ref_long = false;
    }
    // ---- output and removal before decoding (C.5.2.2)
    if (irap && c.no_rasl_output && !first_pic) {
      const bool no_output_of_prior = h.no_output_of_prior_pics;
      if (no_output_of_prior) {
        for (const PicPtr& f : dpb) {
          f->need_output = false;
          f->ref_short = f->ref_long = false;
        }
      } else {
        for (const PicPtr& f : dpb) f->ref_short = f->ref_long = false;
        flush_output(false);
      }
      remove_unused();
    } else {
      remove_unused();
      for (;;) {
        const bool full = static_cast<int>(dpb.size()) >= s.max_dec_pic_buffering[hi] + 1;
        if (!(need_bump(s) || (full && waiting_count() > 0))) break;
        if (!bump_one()) break;
      }
    }
    pop_queue(eff_queue());
    c.poc = poc;
    c.st_before = std::move(st_before);
    c.st_after = std::move(st_after);
    c.lt_curr = std::move(lt_curr);
  }

  void finish_picture() {
    if (!cur) return;
    std::unique_ptr<Cur> own = std::move(cur);
    Cur& c = *own;
    if (c.skip || c.desc.slice_offsets.empty()) return;
    if (!c.prepared) prepare_picture(c);
    const Sps& s = *c.sps;
    const SliceHeader& h = c.first;
    const int poc = c.poc;
    const std::vector<PicPtr>& st_before = c.st_before;
    const std::vector<PicPtr>& st_after = c.st_after;
    const std::vector<PicPtr>& lt_curr = c.lt_curr;
    // the surface of the picture is taken just before it is decoded
    PicPtr fs = std::make_shared<Pic>();
    fs->poc = poc;
    fs->current = true;
    const int idx = alloc_pic_idx();
    if (idx < 0) return;
    fs->pic_idx = idx;
    surf_pic[static_cast<size_t>(idx)] = fs;
    // pictures this one's reference picture set no longer holds give their surfaces back after it has taken its own (measured)
    PicDesc& d = c.desc;
    d.curr_pic_idx = idx;
    d.nal_type = c.nal_type;
    d.sps = c.sps.get();
    d.pps = c.pps.get();
    d.ref_pic = true;
    d.num_delta_pocs_of_ref_rps = h.st_rps_ref_num_delta;
    d.intra_pic = c.all_intra;
    fill_pic_params(c, h);
    PicParams& pp = d.pp;
    pp.curr_poc = poc;
    // slots of the reference picture set: the places the frame stores took when they were stored
    auto slot_of = [&](const PicPtr& f) -> int {
      if (!f) return 0;
      assign_slot(f);
      return std::max(0, f->slot);
    };
    int n_before = 0, n_after = 0, n_lt = 0;
    for (const PicPtr& f : st_before)
      if (n_before < 8) pp.st_before[n_before++] = static_cast<uint8_t>(slot_of(f));
    for (const PicPtr& f : st_after)
      if (n_after < 8) pp.st_after[n_after++] = static_cast<uint8_t>(slot_of(f));
    for (const PicPtr& f : lt_curr)
      if (n_lt < 8) pp.lt_curr[n_lt++] = static_cast<uint8_t>(slot_of(f));
    pp.num_st_before = n_before;
    pp.num_st_after = n_after;
    pp.num_lt_curr = n_lt;
    pp.num_poc_total_curr = h.num_pic_total_curr;
    for (int i = 0; i < 16; ++i) {
      RefSlot& rs = d.slots[i];
      rs = RefSlot{};
      const PicPtr& f = slots[i];
      if (!f || !f->ref_any()) continue;
      rs.pic_idx = f->pic_idx;
      rs.poc = f->poc;
      rs.long_term = f->ref_long;
      pp.refs[i].poc = f->poc;
      pp.refs[i].long_term = f->ref_long;
    }
    if (want_sei && !c.sei.empty()) sink->sei(idx, c.sei);
    const bool show = sink->decode(d) != 0;
    // ---- store the current picture and bump (C.3.4, C.5.2.3)
    fs->current = false;
    fs->ref_short = true;
    const bool out_flag = h.pic_output_flag && !c.no_output;
    if (out_flag) {
      for (const PicPtr& f : dpb)
        if (f->need_output && f->poc > poc) ++f->latency;
      fs->need_output = show;
      fs->latency = 0;
    }
    if (!fs->need_output) {   // never displayed: its surface goes as soon as it is no longer a reference
      fs->displayed = true;
      fs->never_output = true;
      fs->disp_seq = -(1 << 30);
    }
    fs->in_dpb = true;
    fs->progressive = !s.field_seq;
    dpb.push_back(fs);
    last_pic = fs;
    assign_slot(fs);   // a frame store keeps its place in the picture table while it is in the DPB
    while (need_bump(s) && bump_one()) {
    }
    pop_queue(eff_queue());
    remove_unused();
    first_pic = false;
    after_eos = false;
  }

};

HevcParser::HevcParser(ParserSink* sink, unsigned max_decode_surfaces, unsigned clock_rate, unsigned max_display_delay, bool want_sei)
    : p_(new Impl(sink, max_decode_surfaces, clock_rate, max_display_delay, want_sei)) {}
HevcParser::~HevcParser() = default;
void HevcParser::feed(const uint8_t* data, size_t n, bool has_ts, int64_t ts, bool discontinuity, bool end_of_picture) {
  p_->feed(data, n, has_ts, ts, discontinuity, end_of_picture);
}
void HevcParser::end_of_stream() { p_->end_of_stream(); }

}  // namespace vgpu_hevc
