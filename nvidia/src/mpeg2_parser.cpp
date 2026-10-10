// See mpeg2_parser.hpp.
#include "mpeg2_parser.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <queue>

namespace vgpu_mpeg2 {
namespace {

int gcd_int(int a, int b) {
  while (b) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// A frame store of the parser: the decode surface a coded frame (or the pair of fields of one) lives in.
struct Frm {
  int idx = -1;
  int type = 1;                 // of the first field or the frame
  bool ref = false;             // may still be a reference
  bool need_output = false;     // decoded and not yet displayed
  bool displayed = false;
  bool current = false;         // being decoded (the first field has been, the second has not)
  bool released = false;
  bool progressive = true, tff = false;
  int repeat = 0;
};
using FrmPtr = std::shared_ptr<Frm>;

constexpr unsigned kFrameRates[9][2] = {{0, 0}, {24000, 1001}, {24000, 1000}, {25000, 1000}, {30000, 1001}, {30000, 1000}, {50000, 1000}, {60000, 1001}, {60000, 1000}};

}  // namespace

struct Mpeg2Parser::Impl {
  ParserSink* sink;
  unsigned max_surfaces, clock_rate, max_delay;

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
  int64_t last_pic_ts = 0, last_pic_dur = 0;
  bool have_last_ts = false;
  std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>> ts_heap;

  // ---- sequence ----
  SeqHeader seq;
  bool have_seq = false;          // a sequence header has been read
  bool seq_has_ext = false;
  uint8_t intra_matrix[64], inter_matrix[64];
  SeqInfo last_info;
  bool have_info = false;
  bool seq_disabled = false;      // the application's sequence callback refused the sequence: its pictures are not decoded
  bool seq_valid = false;         // the last sequence header is one the card accepts (a frame rate code 1 to 8)
  int pool = 0;
  std::deque<int> free_list;      // decode surfaces not in use, in the order they were released
  std::vector<FrmPtr> surf;       // the frame store each surface holds
  FrmPtr held;                    // the frame displayed last (its surface stays out of the pool until the next display)

  // ---- reference frames and the display queue ----
  FrmPtr ref_old, ref_new;        // the two most recent reference frames, oldest first
  FrmPtr waiting;                 // a reference frame decoded and not yet queued for display (shown once the next reference frame is decoded)
  std::deque<FrmPtr> out_queue;   // frames queued for display; the application's display delay is how many it holds back
  FrmPtr pending_first;           // a first field whose second field has not come
  int pending_parity = 0;
  GopHeader gop;

  // ---- the picture being assembled ----
  struct Cur {
    PicHeader h;
    bool skip = false;
    PicDesc desc;
    int64_t ts = 0;
    uint64_t offset = 0;
    bool has_slices = false;
  };
  std::unique_ptr<Cur> cur;

  Impl(ParserSink* s, unsigned maxs, unsigned clock, unsigned delay) : sink(s), max_surfaces(std::max(1u, maxs)), clock_rate(clock ? clock : 10000000), max_delay(delay) {
    default_matrices(intra_matrix, inter_matrix);
  }

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
    flush_display();
  }

  void process(bool eos) {
    for (;;) {
      const int s1 = find_start(buf, 0);
      if (s1 < 0) {
        if (buf.size() > 2) {
          buf_base += buf.size() - 2;
          buf.erase(buf.begin(), buf.end() - 2);
        }
        return;
      }
      // measured: the card's parser acts on the start code of a picture, group of pictures, sequence header or sequence end as soon as
      // it has read it -- the picture before it is complete then -- not when the unit it starts is complete
      if (!start_seen && buf.size() >= static_cast<size_t>(s1) + 4) {
        start_seen = true;
        const int code = buf[static_cast<size_t>(s1) + 3];
        if (code == 0x00 || code == 0xB3 || code == 0xB8 || code == 0xB7) finish_picture();
      }
      const int s2 = find_start(buf, static_cast<size_t>(s1) + 3);
      if (s2 >= 0) {
        consume_unit(buf.data() + s1 + 3, static_cast<size_t>(s2) - (static_cast<size_t>(s1) + 3), buf_base + static_cast<uint64_t>(s1));
        buf_base += static_cast<uint64_t>(s2);
        buf.erase(buf.begin(), buf.begin() + s2);
        start_seen = false;
        continue;
      }
      if (eos) {
        const size_t end = buf.size();
        if (end > static_cast<size_t>(s1) + 3) consume_unit(buf.data() + s1 + 3, end - (static_cast<size_t>(s1) + 3), buf_base + static_cast<uint64_t>(s1));
        buf_base += buf.size();
        buf.clear();
        start_seen = false;
      }
      return;
    }
  }
  bool start_seen = false;

  // ================================================================== start code units
  void consume_unit(const uint8_t* u, size_t n, uint64_t abs_offset) {
    if (n < 1) return;
    const int code = u[0];
    const uint8_t* d = u + 1;
    const size_t dn = n - 1;
    if (code >= 0x01 && code <= 0xAF) {
      consume_slice(u, n);
      return;
    }
    switch (code) {
      case 0x00:
        finish_picture();
        begin_picture(d, dn, abs_offset);
        break;
      case 0xB3: {
        finish_picture();
        SeqHeader s;
        if (!parse_sequence_header(d, dn, &s)) break;
        seq = s;
        have_seq = true;
        seq_valid = s.frame_rate_code >= 1 && s.frame_rate_code <= 8;
        if (!s.has_extension) s.progressive_sequence = 1;
        std::memcpy(intra_matrix, seq.intra_matrix, 64);
        std::memcpy(inter_matrix, seq.inter_matrix, 64);
        break;
      }
      case 0xB5:
        if (dn < 1) break;
        switch (d[0] >> 4) {
          case 1:
            if (have_seq) parse_sequence_extension(d, dn, &seq);
            break;
          case 2:
            if (have_seq) parse_sequence_display_extension(d, dn, &seq);
            break;
          case 3: {
            bool li = false, lni = false;
            if (parse_quant_matrix_extension(d, dn, intra_matrix, inter_matrix, &li, &lni)) {
            }
            break;
          }
          case 8:
            if (cur && !cur->has_slices) parse_picture_coding_extension(d, dn, &cur->h);
            break;
          default:
            break;
        }
        break;
      case 0xB8:
        finish_picture();
        parse_gop_header(d, dn, &gop);
        break;
      case 0xB7:
        finish_picture();   // measured: the pictures waiting for display stay until the next sequence is announced
        break;
      default:
        break;
    }
  }

  void consume_slice(const uint8_t* u, size_t n) {
    if (!cur || cur->skip) return;
    // the sequence is announced when the first slice of an intra picture has been read (the picture is not complete yet), also when a
    // new sequence header changed it
    if (!cur->has_slices) {
      const int structure = cur->h.has_extension ? cur->h.picture_structure : 3;
      const bool second = pending_first && structure != 3 && (structure == 2 ? 1 : 0) != pending_parity;
      assign_timestamp(*cur, second);
      if (cur->h.picture_coding_type == 1 && have_seq && seq_valid && !announce_sequence()) {
        cur->skip = true;
        return;
      }
    }
    PicDesc& dsc = cur->desc;
    dsc.slice_offsets.push_back(static_cast<unsigned>(dsc.data.size()));
    dsc.data.insert(dsc.data.end(), {0, 0, 1});
    dsc.data.insert(dsc.data.end(), u, u + n);
    cur->has_slices = true;
  }

  // ================================================================== pictures
  void begin_picture(const uint8_t* d, size_t n, uint64_t abs_offset) {
    cur = std::make_unique<Cur>();
    cur->offset = abs_offset;
    if (!parse_picture_header(d, n, &cur->h) || !have_seq) cur->skip = true;
  }

  // The timestamp of the picture, taken when its first slice is read (the picture coding extension has said by then whether it is a
  // field): the last timestamp given at or before the picture header, once; a picture without one follows the last by a frame
  // duration, and the second field of a frame has its first field's.
  void assign_timestamp(Cur& c, bool second_field) {
    int64_t ts = 0;
    bool taken = false;
    for (auto it = marks.rbegin(); it != marks.rend(); ++it)
      if (it->at <= c.offset) {
        if (!it->used) {
          it->used = true;
          ts = it->ts;
          taken = true;
        }
        break;
      }
    if (!taken) {
      if (have_last_ts) ts = second_field ? last_pic_ts : last_pic_ts + last_pic_dur;
      else ts = 0;
    }
    while (marks.size() > 1 && marks[1].at <= c.offset) marks.pop_front();
    if (!second_field || taken) {
      last_pic_ts = ts;
      have_last_ts = true;
      // how long the picture lasts (measured): a frame period, the period doubled or tripled by a progressive sequence's repeat
      // flags, and half a period more by an interlaced one's
      const unsigned num = seq_fps_num() ? seq_fps_num() : 30, den = seq_fps_num() ? seq_fps_den() : 1;
      const int64_t period = static_cast<int64_t>(clock_rate) * den / num;
      const int rep = repeat_of(c.h);
      last_pic_dur = rep == 2 ? 2 * period : rep == 4 ? 3 * period : rep == 1 ? period + period / 2 : period;
    }
    c.ts = ts;
  }

  unsigned seq_fps_num() const {
    if (seq.frame_rate_code < 1 || seq.frame_rate_code > 8) return 0;
    return kFrameRates[seq.frame_rate_code][0];
  }
  unsigned seq_fps_den() const {
    if (seq.frame_rate_code < 1 || seq.frame_rate_code > 8) return 0;
    return kFrameRates[seq.frame_rate_code][1];
  }

  SeqInfo make_info() const {
    SeqInfo i;
    i.coded_w = seq.horizontal_size;   // measured: not rounded up
    i.coded_h = (seq.progressive_sequence || !seq.has_extension) ? (seq.vertical_size + 15) & ~15 : (seq.vertical_size + 31) & ~31;
    i.disp_left = 0;
    i.disp_top = 0;
    i.disp_right = seq.horizontal_size;
    i.disp_bottom = seq.vertical_size;
    i.fps_num = seq_fps_num();
    i.fps_den = seq_fps_den();
    if (i.fps_num && i.fps_den) {
      const unsigned g = static_cast<unsigned>(gcd_int(static_cast<int>(i.fps_num), static_cast<int>(i.fps_den)));
      (void)g;   // the card reports the table's numbers unreduced (30000/1000)
    }
    i.mpeg1 = !seq.has_extension;
    i.progressive = i.mpeg1 || seq.progressive_sequence != 0;
    i.chroma_format = seq.has_extension ? seq.chroma_format : 1;
    i.bitrate = seq.bit_rate * 400;
    i.min_surfaces = 4;
    long dx = seq.horizontal_size, dy = seq.vertical_size;
    switch (seq.aspect_ratio_information) {
      case 2: dx = 4; dy = 3; break;
      case 3: dx = 16; dy = 9; break;
      case 4: dx = 221; dy = 100; break;
      default: break;
    }
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
    if (i.mpeg1) {
      i.video_format = 5;   // measured: the card's defaults for a stream without sequence extension (MPEG-1): unspecified
      i.primaries = i.transfer = i.matrix = 2;
    }
    if (seq.has_display) {
      i.video_format = seq.video_format;
      i.primaries = seq.colour_primaries;
      i.transfer = seq.transfer_characteristics;
      i.matrix = seq.matrix_coefficients;
    }
    i.supported = i.chroma_format == 1 && !i.mpeg1;
    return i;
  }

  static bool same_info(const SeqInfo& a, const SeqInfo& b) {
    return a.coded_w == b.coded_w && a.coded_h == b.coded_h && a.disp_right == b.disp_right && a.disp_bottom == b.disp_bottom && a.fps_num == b.fps_num &&
           a.fps_den == b.fps_den && a.progressive == b.progressive && a.chroma_format == b.chroma_format && a.bitrate == b.bitrate && a.dar_x == b.dar_x &&
           a.dar_y == b.dar_y && a.video_format == b.video_format && a.primaries == b.primaries && a.transfer == b.transfer && a.matrix == b.matrix;
  }

  // ---- surfaces ----
  // The pool is a queue of free surface indices: the one that was released first is handed out first. A surface goes back to the pool
  // when its frame is neither a reference nor waiting for display -- except the frame displayed last: the application may still be
  // reading it, so it stays out until the next frame is displayed (or until the pool runs dry and it is taken back).
  void try_free(const FrmPtr& f) {
    if (!f || f->released || f->idx < 0 || f->ref || f->need_output || f->current || f == held) return;
    f->released = true;
    free_list.push_back(f->idx);
  }

  int alloc_idx() {
    for (;;) {
      if (!free_list.empty()) {
        const int i = free_list.front();
        free_list.pop_front();
        return i;
      }
      if (held && !held->ref && !held->need_output && !held->current) {
        const int i = held->idx;
        held->released = true;
        held.reset();
        return i;
      }
      // every surface is in use: show pictures until one is free
      if (!out_queue.empty() && !out_queue.front()->current) {
        display_front();
      } else if (waiting) {
        queue_waiting();
      } else {
        return -1;
      }
    }
  }

  void display_frame(const FrmPtr& f) {
    DisplayInfo di;
    di.pic_idx = f->idx;
    di.progressive_frame = f->progressive;
    di.top_field_first = f->tff;
    di.repeat_first_field = f->repeat;
    di.timestamp = 0;
    if (!ts_heap.empty()) {
      di.timestamp = ts_heap.top();
      ts_heap.pop();
    }
    f->need_output = false;
    sink->display(di);
    f->displayed = true;
    FrmPtr prev = held;
    held = f;
    if (prev) try_free(prev);
    try_free(f);
  }

  // The frames shown are held back by the application's display delay (at most two pictures: measured; more than that is
  // not possible with four decode surfaces).
  // With field pictures the card keeps the same delay at most one picture (measured; two or three with fields is not reproduced).
  size_t delay_limit() const { return std::min<size_t>(max_delay, saw_field ? 1 : 2); }
  bool saw_field = false;

  void display_front() {
    FrmPtr f = out_queue.front();
    out_queue.pop_front();
    display_frame(f);
  }

  void queue_frame(const FrmPtr& f) {
    out_queue.push_back(f);
    pop_queue(false);
  }

  // Shows the frames beyond the delay. The front must be complete; when a field picture starts, a B frame in front is not shown yet.
  void pop_queue(bool first_field) {
    while (out_queue.size() > delay_limit() && !out_queue.front()->current &&
           !(first_field && out_queue.front()->type == 3))
      display_front();
  }

  // the reference frame waiting for the next one joins the display queue
  void queue_waiting() {
    if (!waiting) return;
    FrmPtr f = waiting;
    waiting.reset();
    queue_frame(f);
  }

  void flush_display() {
    queue_waiting();
    while (!out_queue.empty()) display_front();
  }

  // ---- finishing a picture: the decode callback and what follows ----
  // The sequence is announced when the first slice of an intra picture has been read (the picture is not complete yet) and it differs
  // from the last one announced in any field. Returns false if the application's callback refused it: its pictures are dropped until the
  // next sequence that differs. A change of the coded size ends the old sequence: the pictures waiting for display are shown first.
  bool announce_sequence() {
    const SeqInfo info = make_info();
    const bool first = !have_info;
    const bool changed = first || !same_info(info, last_info);
    if (!changed) return !seq_disabled;
    const bool geometry = !first && (info.coded_w != last_info.coded_w || info.coded_h != last_info.coded_h);
    if (geometry) flush_display();
    last_info = info;
    have_info = true;
    const int ret = sink->sequence(info);
    seq_disabled = ret == 0;
    pool = ret > 1 ? ret : std::max(1, info.min_surfaces);
    if (first || geometry) {
      // the frames of the old sequence are all output; the surfaces keep their state, in the order they are handed out again
      if (ref_old) ref_old->ref = false;
      if (ref_new) ref_new->ref = false;
      ref_old.reset();
      ref_new.reset();
      pending_first.reset();
      held.reset();
      std::deque<int> order;
      std::vector<bool> in(static_cast<size_t>(pool), false);
      for (int i : free_list)
        if (i < pool && !in[static_cast<size_t>(i)]) {
          order.push_back(i);
          in[static_cast<size_t>(i)] = true;
        }
      for (int i = 0; i < pool; ++i)
        if (!in[static_cast<size_t>(i)]) order.push_back(i);
      free_list = order;
      surf.assign(static_cast<size_t>(pool), FrmPtr());
    } else if (static_cast<int>(surf.size()) != pool) {
      surf.resize(static_cast<size_t>(pool));
    }
    return !seq_disabled;
  }

  void finish_picture() {
    if (!cur) return;
    std::unique_ptr<Cur> own = std::move(cur);
    Cur& c = *own;
    if (c.skip || !c.has_slices || !have_seq || !seq_valid || seq_disabled || !have_info) return;
    const PicHeader& h = c.h;
    const int type = h.picture_coding_type;
    if (type < 1 || type > 3) return;
    const int structure = h.has_extension ? h.picture_structure : 3;
    const bool field = structure != 3;
    const int parity = structure == 2 ? 1 : 0;
    // ---- which pictures the card decodes
    bool second = false;
    if (pending_first) {
      if (!field) return;   // a frame picture while a field is waiting for its partner is not decoded
      if (parity != pending_parity) {
        second = true;
      } else {
        // a first field of the same parity again: it replaces the waiting one
        pending_first->current = false;
        const FrmPtr old = pending_first;
        pending_first.reset();
        reuse_idx_ = old->idx;
        old->ref = false;
        old->need_output = false;
        out_queue.erase(std::remove(out_queue.begin(), out_queue.end(), old), out_queue.end());
        old->released = true;   // the surface is taken again below
      }
    }
    if (!second) {
      if (type == 2 && !ref_new) return;
      if (type == 3 && !(ref_old && ref_new)) return;
      if (!have_info && type != 1) return;
    }
    // ---- the frame store
    FrmPtr f;
    if (second) {
      f = pending_first;
    } else {
      f = std::make_shared<Frm>();
      int idx;
      if (reuse_idx_ >= 0) {
        idx = reuse_idx_;
        reuse_idx_ = -1;
      } else {
        idx = alloc_idx();
      }
      if (idx < 0) return;
      f->idx = idx;
      f->type = type;
      f->current = true;
      f->progressive = h.has_extension ? h.progressive_frame : true;
      f->tff = h.top_field_first;
      f->repeat = repeat_of(h);
      surf[static_cast<size_t>(idx)] = f;
      ts_heap.push(c.ts);
    }
    PicDesc& d = c.desc;
    d.curr_pic_idx = f->idx;
    d.field_pic = field;
    d.bottom_field = field && parity == 1;
    d.second_field = second;
    d.ref_pic = type != 3;
    d.intra_pic = type == 1;
    d.hdr = h;
    if (type != 3) {
      // measured: an intra picture too names the reference frame before it; the second field of a frame names the frame itself
      d.fwd_idx = ref_new ? ref_new->idx : -1;
      d.bwd_idx = second ? f->idx : -1;
    } else {
      d.fwd_idx = ref_old->idx;
      d.bwd_idx = ref_new->idx;
    }
    fill_pic_params(c, structure);
    sink->decode(d);
    // ---- what the picture changes
    if (field && !second) {
      pending_first = f;
      pending_parity = parity;
      saw_field = true;
      // a field picture takes part in the display queue from its first field on (measured)
      if (delay_limit() > 0) {
        if (type == 3) {
          out_queue.push_back(f);
        } else if (waiting) {
          FrmPtr w = waiting;
          waiting.reset();
          out_queue.push_back(w);
        }
      }
      pop_queue(true);
      return;
    }
    pending_first.reset();
    f->current = false;
    f->need_output = true;
    if (type == 3) {
      if (std::find(out_queue.begin(), out_queue.end(), f) != out_queue.end()) pop_queue(false);
      else queue_frame(f);
    } else if (seq.low_delay) {
      // a low delay sequence has no reordering: the picture is shown at once
      queue_waiting();
      f->ref = true;
      if (ref_old) {
        ref_old->ref = false;
        try_free(ref_old);
      }
      ref_old = ref_new;
      ref_new = f;
      queue_frame(f);
    } else {
      f->ref = true;
      if (ref_old) {
        ref_old->ref = false;
        try_free(ref_old);
      }
      ref_old = ref_new;
      ref_new = f;
      queue_waiting();
      waiting = f;
      pop_queue(false);
    }
  }

  int reuse_idx_ = -1;

  // CUVIDPARSERDISPINFO::repeat_first_field (measured): 0 for a picture whose first field is not repeated; for a repeated one 1 in an
  // interlaced sequence (one more field) and 2 or 4 in a progressive one (the frame shown twice, or three times when the top field is first)
  int repeat_of(const PicHeader& h) const {
    if (!h.has_extension || !h.repeat_first_field) return 0;
    if (!seq.progressive_sequence) return 1;
    return h.top_field_first ? 4 : 2;
  }

  void fill_pic_params(Cur& c, int structure) {
    const PicHeader& h = c.h;
    PicDesc& d = c.desc;
    PicParams& pp = d.pp;
    d.pic_width_in_mbs = seq.horizontal_size / 16;   // measured: the card's value is the width divided by 16, rounded down
    pp.mb_width = (seq.horizontal_size + 15) / 16;
    pp.mb_height = seq.progressive_sequence ? (seq.vertical_size + 15) / 16 : 2 * ((seq.vertical_size + 31) / 32);
    pp.field_pic = structure != 3;
    pp.bottom_field = structure == 2;
    pp.second_field = d.second_field;
    pp.picture_coding_type = h.picture_coding_type;
    pp.full_pel_forward = h.full_pel_forward;
    pp.full_pel_backward = h.full_pel_backward;
    for (int s = 0; s < 2; ++s)
      for (int t = 0; t < 2; ++t) pp.f_code[s][t] = h.f_code[s][t];
    pp.intra_dc_precision = h.intra_dc_precision;
    pp.frame_pred_frame_dct = h.frame_pred_frame_dct;
    pp.concealment_motion_vectors = h.concealment_motion_vectors;
    pp.q_scale_type = h.q_scale_type;
    pp.intra_vlc_format = h.intra_vlc_format;
    pp.alternate_scan = h.alternate_scan;
    pp.top_field_first = h.top_field_first;
    std::memcpy(pp.intra_matrix, intra_matrix, 64);
    std::memcpy(pp.inter_matrix, inter_matrix, 64);
  }
};

Mpeg2Parser::Mpeg2Parser(ParserSink* sink, unsigned max_decode_surfaces, unsigned clock_rate, unsigned max_display_delay)
    : p_(new Impl(sink, max_decode_surfaces, clock_rate, max_display_delay)) {}
Mpeg2Parser::~Mpeg2Parser() = default;
void Mpeg2Parser::feed(const uint8_t* data, size_t n, bool has_ts, int64_t ts, bool discontinuity, bool end_of_picture) {
  p_->feed(data, n, has_ts, ts, discontinuity, end_of_picture);
}
void Mpeg2Parser::end_of_stream() { p_->end_of_stream(); }

}  // namespace vgpu_mpeg2
