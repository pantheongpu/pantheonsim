// Part of h264_decode.cpp: picture and slice level.

bool Dec::run(const std::vector<SliceData>& slices, std::string* err) {
  bool ok = true;
  field_pic_ = pp_.field_pic != 0;
  bottom_ = pp_.bottom_field != 0;
  mbaff_ = pp_.mbaff != 0 && !field_pic_;
  if (pp_.chroma_format_idc != 1 || pp_.bit_depth_luma != 8 || pp_.bit_depth_chroma != 8) {
    *err = "only 8-bit 4:2:0 pictures are decoded";
    return false;
  }
  if (pp_.transform_bypass) {
    *err = "lossless (transform bypass) coding is not decoded";
    return false;
  }
  if (pp_.num_slice_groups > 1) {
    *err = "slice groups (flexible macroblock ordering) are not decoded";
    return false;
  }
  if (pp_.mbs_w <= 0 || pp_.mbs_h <= 0 || (field_pic_ && (pp_.mbs_h & 1))) {
    *err = "bad picture size";
    return false;
  }
  W_ = pp_.mbs_w;
  H_ = field_pic_ ? pp_.mbs_h / 2 : pp_.mbs_h;
  nmb_ = W_ * H_;
  max_frame_num_ = 1 << pp_.log2_max_frame_num;
  if (cur_->mbs_w != pp_.mbs_w || cur_->mbs_h != pp_.mbs_h || cur_->y.empty()) cur_->alloc(pp_.mbs_w, pp_.mbs_h);
  auto pd = std::make_shared<PicData>();
  pd->mbs_w = W_;
  pd->mbs_h = H_;
  pd->field_pic = field_pic_;
  pd->bottom = bottom_;
  pd->mbaff = mbaff_;
  pd->poc[0] = pp_.poc[0];
  pd->poc[1] = pp_.poc[1];
  pd->mb.assign(static_cast<size_t>(nmb_), MbInfo());
  if (field_pic_) {
    cur_->pic[1 + (bottom_ ? 1 : 0)] = pd;
    cur_->coded_as_frame = false;
  } else {
    cur_->pic[0] = pd;
    cur_->pic[1].reset();
    cur_->pic[2].reset();
    cur_->coded_as_frame = true;
  }
  pd_ = pd.get();
  cur_->poc[0] = pp_.poc[0];
  cur_->poc[1] = pp_.poc[1];
  const int fs = cur_->stride_y, cs = cur_->stride_c;
  py_ = {cur_->y.data() + (field_pic_ && bottom_ ? fs : 0), field_pic_ ? 2 * fs : fs, W_ * 16, H_ * 16};
  pu_ = {cur_->u.data() + (field_pic_ && bottom_ ? cs : 0), field_pic_ ? 2 * cs : cs, W_ * 8, H_ * 8};
  pv_ = {cur_->v.data() + (field_pic_ && bottom_ ? cs : 0), field_pic_ ? 2 * cs : cs, W_ * 8, H_ * 8};
  cur_poc_ = field_pic_ ? pp_.poc[bottom_ ? 1 : 0] : std::min(pp_.poc[0], pp_.poc[1]);
  for (const SliceData& sd : slices)
    if (!decode_slice(sd)) ok = false;
  int missing = 0;
  for (int i = 0; i < nmb_; ++i)
    if (pd_->mb[i].slice < 0) ++missing;
  if (missing) {
    ok = false;
    if (err_.empty()) err_ = "macroblocks missing from the picture";
  }
  deblock_picture();
  if (!ok) *err = err_;
  return ok;
}

// ---- neighbour derivation (6.4.8 - 6.4.12) ------------------------------------------------------
Nb Dec::locate(int xN, int yN, int maxW, int maxH) const {
  Nb r;
  if (yN > maxH - 1) return r;
  if (xN > maxW - 1 && yN >= 0) return r;
  if (!mbaff_) {
    int a;
    if (xN < 0) a = yN < 0 ? nb_d_ : mbaddr_a_;
    else if (xN < maxW) a = yN < 0 ? mbaddr_b_ : addr_;
    else a = nb_c_;
    if (a < 0) return r;
    r.addr = a;
    r.x = (xN + maxW) % maxW;
    r.y = (yN + maxH) % maxH;
    return r;
  }
  const bool top = (addr_ & 1) == 0;
  const bool cf = !mb_field_;
  int a = -1, yM = 0;
  if (xN < 0) {
    if (yN < 0) {
      if (cf) {
        if (top) {
          if (pair_d_ >= 0) {
            a = pair_d_ + 1;
            yM = yN;
          }
        } else if (pair_a_ >= 0) {
          if (!mbi(pair_a_).field) {
            a = pair_a_;
            yM = yN;
          } else {
            a = pair_a_ + 1;
            yM = (yN + maxH) >> 1;
          }
        }
      } else {
        if (top) {
          if (pair_d_ >= 0) {
            if (!mbi(pair_d_).field) {
              a = pair_d_ + 1;
              yM = 2 * yN;
            } else {
              a = pair_d_;
              yM = yN;
            }
          }
        } else if (pair_d_ >= 0) {
          a = pair_d_ + 1;
          yM = yN;
        }
      }
    } else {
      if (pair_a_ < 0) return r;
      const bool af = !mbi(pair_a_).field;
      if (cf) {
        if (top) {
          if (af) {
            a = pair_a_;
            yM = yN;
          } else {
            a = pair_a_ + (yN & 1);
            yM = yN >> 1;
          }
        } else {
          if (af) {
            a = pair_a_ + 1;
            yM = yN;
          } else {
            a = pair_a_ + (yN & 1);
            yM = (yN + maxH) >> 1;
          }
        }
      } else {
        if (top) {
          if (af) {
            if (yN < maxH / 2) {
              a = pair_a_;
              yM = yN << 1;
            } else {
              a = pair_a_ + 1;
              yM = (yN << 1) - maxH;
            }
          } else {
            a = pair_a_;
            yM = yN;
          }
        } else {
          if (af) {
            if (yN < maxH / 2) {
              a = pair_a_;
              yM = (yN << 1) + 1;
            } else {
              a = pair_a_ + 1;
              yM = (yN << 1) + 1 - maxH;
            }
          } else {
            a = pair_a_ + 1;
            yM = yN;
          }
        }
      }
    }
  } else if (xN < maxW) {
    if (yN < 0) {
      if (cf) {
        if (top) {
          if (pair_b_ >= 0) {
            a = pair_b_ + 1;
            yM = yN;
          }
        } else {
          a = addr_ - 1;
          yM = yN;
        }
      } else {
        if (top) {
          if (pair_b_ >= 0) {
            if (!mbi(pair_b_).field) {
              a = pair_b_ + 1;
              yM = 2 * yN;
            } else {
              a = pair_b_;
              yM = yN;
            }
          }
        } else if (pair_b_ >= 0) {
          a = pair_b_ + 1;
          yM = yN;
        }
      }
    } else {
      a = addr_;
      yM = yN;
    }
  } else {
    if (cf) {
      if (top && pair_c_ >= 0) {
        a = pair_c_ + 1;
        yM = yN;
      }
    } else {
      if (top) {
        if (pair_c_ >= 0) {
          if (!mbi(pair_c_).field) {
            a = pair_c_ + 1;
            yM = 2 * yN;
          } else {
            a = pair_c_;
            yM = yN;
          }
        }
      } else if (pair_c_ >= 0) {
        a = pair_c_ + 1;
        yM = yN;
      }
    }
  }
  if (a < 0) return r;
  r.addr = a;
  r.x = (xN + maxW) % maxW;
  r.y = (yM + maxH) % maxH;
  return r;
}

uint8_t* Dec::luma_at(int addr, int x, int y) const {
  if (!mbaff_) {
    const int mx = addr % W_, my = addr / W_;
    return py_.p + static_cast<ptrdiff_t>(my * 16 + y) * py_.stride + mx * 16 + x;
  }
  const int pair = addr >> 1, px = pair % W_, pyr = pair / W_;
  const bool fld = pd_->mb[addr].field;
  const int row = fld ? pyr * 32 + (addr & 1) + 2 * y : pyr * 32 + (addr & 1) * 16 + y;
  return py_.p + static_cast<ptrdiff_t>(row) * py_.stride + px * 16 + x;
}

uint8_t* Dec::chroma_at(int plane, int addr, int x, int y) const {
  const Plane& p = plane == 0 ? pu_ : pv_;
  if (!mbaff_) {
    const int mx = addr % W_, my = addr / W_;
    return p.p + static_cast<ptrdiff_t>(my * 8 + y) * p.stride + mx * 8 + x;
  }
  const int pair = addr >> 1, px = pair % W_, pyr = pair / W_;
  const bool fld = pd_->mb[addr].field;
  const int row = fld ? pyr * 16 + (addr & 1) + 2 * y : pyr * 16 + (addr & 1) * 8 + y;
  return p.p + static_cast<ptrdiff_t>(row) * p.stride + px * 8 + x;
}

// Prepares the macroblock at `addr` for decoding: its record, its place in the picture and its neighbours.
// In a macroblock-adaptive frame, mb_field_ (the pair's field decoding flag) must be set first.
void Dec::setup_mb(int addr) {
  addr_ = addr;
  mb_ = &pd_->mb[addr];
  *mb_ = MbInfo();
  mb_->slice = static_cast<int16_t>(slice_id_);
  if (!mbaff_) {
    mbx_ = addr % W_;
    mby_ = addr / W_;
    mb_field_ = field_pic_;
    mb_->field = field_pic_;
    org_x_ = mbx_ * 16;
    org_y_ = mby_ * 16;
    ly_ = py_.p + static_cast<ptrdiff_t>(mby_ * 16) * py_.stride + mbx_ * 16;
    lu_ = pu_.p + static_cast<ptrdiff_t>(mby_ * 8) * pu_.stride + mbx_ * 8;
    lv_ = pv_.p + static_cast<ptrdiff_t>(mby_ * 8) * pv_.stride + mbx_ * 8;
    ls_ = py_.stride;
    lcs_ = pu_.stride;
    mbaddr_a_ = (mbx_ > 0 && avail(addr - 1)) ? addr - 1 : -1;
    mbaddr_b_ = (mby_ > 0 && avail(addr - W_)) ? addr - W_ : -1;
    nb_c_ = (mby_ > 0 && mbx_ < W_ - 1 && avail(addr - W_ + 1)) ? addr - W_ + 1 : -1;
    nb_d_ = (mby_ > 0 && mbx_ > 0 && avail(addr - W_ - 1)) ? addr - W_ - 1 : -1;
    return;
  }
  // macroblock-adaptive frame/field: the macroblocks of a pair are the top and bottom halves of a frame pair, or the
  // top-field and bottom-field rows of a field pair (6.4.1)
  const int pair = addr >> 1, bot = addr & 1;
  mbx_ = pair % W_;
  mby_ = pair / W_;
  mb_->field = mb_field_;
  pair_a_ = (mbx_ > 0 && avail(2 * (pair - 1))) ? 2 * (pair - 1) : -1;
  pair_b_ = (mby_ > 0 && avail(2 * (pair - W_))) ? 2 * (pair - W_) : -1;
  pair_c_ = (mby_ > 0 && mbx_ < W_ - 1 && avail(2 * (pair - W_ + 1))) ? 2 * (pair - W_ + 1) : -1;
  pair_d_ = (mby_ > 0 && mbx_ > 0 && avail(2 * (pair - W_ - 1))) ? 2 * (pair - W_ - 1) : -1;
  const int y0 = mb_field_ ? mby_ * 32 + bot : mby_ * 32 + bot * 16;
  const int cy0 = mb_field_ ? mby_ * 16 + bot : mby_ * 16 + bot * 8;
  ls_ = mb_field_ ? 2 * py_.stride : py_.stride;
  lcs_ = mb_field_ ? 2 * pu_.stride : pu_.stride;
  ly_ = py_.p + static_cast<ptrdiff_t>(y0) * py_.stride + mbx_ * 16;
  lu_ = pu_.p + static_cast<ptrdiff_t>(cy0) * pu_.stride + mbx_ * 8;
  lv_ = pv_.p + static_cast<ptrdiff_t>(cy0) * pv_.stride + mbx_ * 8;
  org_x_ = mbx_ * 16;
  org_y_ = mb_field_ ? mby_ * 16 : mby_ * 32 + bot * 16;
  mbaddr_a_ = locate_l(-1, 0).addr;
  mbaddr_b_ = locate_l(0, -1).addr;
  nb_c_ = nb_d_ = -1;
}

// 7.4.4: a pair whose mb_field_decoding_flag is not (yet) known takes the flag of the pair to the left, else the pair above
// (in the same slice), else it is a frame pair.
bool Dec::infer_pair_field(int addr) const {
  const int pair = addr >> 1, px = pair % W_, py = pair / W_;
  if (px > 0 && avail(2 * (pair - 1))) return mbi(2 * (pair - 1)).field;
  if (py > 0 && avail(2 * (pair - W_))) return mbi(2 * (pair - W_)).field;
  return false;
}

// ---- reference picture lists (8.2.4) --------------------------------------------------------------------
bool Dec::build_ref_lists() {
  ref_[0].clear();
  ref_[1].clear();
  num_ref_[0] = num_ref_[1] = 0;
  if (sh_.is_intra()) return true;
  const bool is_b = sh_.is_b();
  const int max_pic_num = field_pic_ ? 2 * max_frame_num_ : max_frame_num_;
  const int curr_pic_num = field_pic_ ? 2 * pp_.frame_num + 1 : pp_.frame_num;

  std::vector<RefPic> st, lt;   // short-term frames or fields, long-term frames or fields
  auto make = [&](const DpbEntry& e, int structure) {
    RefPic r;
    r.frame = e.frame;
    r.structure = structure;
    r.long_term = e.long_term;
    const int wrap = (!e.long_term && e.frame_num > pp_.frame_num) ? e.frame_num - max_frame_num_ : e.frame_num;
    if (structure == 0) {
      r.poc = std::min(e.poc[0], e.poc[1]);
      r.fn_wrap = wrap;
    } else {
      r.poc = e.poc[structure - 1];
      r.fn_wrap = e.long_term ? e.frame_num * 2 + 1 : 2 * wrap + 1;   // same parity; opposite parity adjusted by the caller
    }
    r.id = e.frame ? e.frame->uid * 4 + structure : -1;
    return r;
  };

  if (!field_pic_) {
    for (const DpbEntry& e : pp_.dpb) {
      if (e.used != 3) continue;
      (e.long_term ? lt : st).push_back(make(e, 0));
    }
    if (!is_b) {
      std::stable_sort(st.begin(), st.end(), [](const RefPic& a, const RefPic& b) { return a.fn_wrap > b.fn_wrap; });
      std::stable_sort(lt.begin(), lt.end(), [](const RefPic& a, const RefPic& b) { return a.fn_wrap < b.fn_wrap; });
      ref_[0] = st;
      ref_[0].insert(ref_[0].end(), lt.begin(), lt.end());
    } else {
      std::vector<RefPic> before, after;
      for (const RefPic& r : st) (r.poc < cur_poc_ ? before : after).push_back(r);
      std::stable_sort(before.begin(), before.end(), [](const RefPic& a, const RefPic& b) { return a.poc > b.poc; });
      std::stable_sort(after.begin(), after.end(), [](const RefPic& a, const RefPic& b) { return a.poc < b.poc; });
      std::stable_sort(lt.begin(), lt.end(), [](const RefPic& a, const RefPic& b) { return a.fn_wrap < b.fn_wrap; });
      ref_[0] = before;
      ref_[0].insert(ref_[0].end(), after.begin(), after.end());
      ref_[0].insert(ref_[0].end(), lt.begin(), lt.end());
      ref_[1] = after;
      ref_[1].insert(ref_[1].end(), before.begin(), before.end());
      ref_[1].insert(ref_[1].end(), lt.begin(), lt.end());
    }
  } else {
    // 8.2.4.2.2, 8.2.4.2.4 and 8.2.4.2.5: frames ordered, then fields picked alternating in parity.
    const int par = bottom_ ? 1 : 0;   // parity of the current field: 0 top, 1 bottom
    struct FrameRef {
      const DpbEntry* e;
      int key;
    };
    auto alternate = [&](std::vector<FrameRef> frames, bool long_term) {
      // fields of the frames in `frames` (in order), same parity first, taking each next field of the wanted parity that is marked
      std::vector<RefPic> out;
      size_t i[2] = {0, 0};
      int want = par;
      auto next_field = [&](int p) -> const FrameRef* {
        while (i[p] < frames.size()) {
          const FrameRef& f = frames[i[p]++];
          const int mask = f.e->used;
          if (mask & (1 << p)) return &f;
        }
        return nullptr;
      };
      for (;;) {
        const FrameRef* f = next_field(want);
        if (!f) {
          // no more fields of this parity: append the remaining fields of the other parity
          for (const FrameRef* g = next_field(1 - want); g; g = next_field(1 - want)) {
            RefPic r = make(*g->e, 1 + (1 - want));
            r.fn_wrap = long_term ? g->e->frame_num * 2 + ((1 - want) == par ? 1 : 0) : 2 * ((g->e->frame_num > pp_.frame_num) ? g->e->frame_num - max_frame_num_ : g->e->frame_num) + ((1 - want) == par ? 1 : 0);
            out.push_back(r);
          }
          break;
        }
        RefPic r = make(*f->e, 1 + want);
        r.fn_wrap = long_term ? f->e->frame_num * 2 + (want == par ? 1 : 0) : 2 * ((f->e->frame_num > pp_.frame_num) ? f->e->frame_num - max_frame_num_ : f->e->frame_num) + (want == par ? 1 : 0);
        out.push_back(r);
        want = 1 - want;
      }
      return out;
    };
    std::vector<FrameRef> sf, lf;
    for (const DpbEntry& e : pp_.dpb) {
      if (e.used == 0) continue;
      if (e.long_term) lf.push_back({&e, e.frame_num});
      else sf.push_back({&e, (e.frame_num > pp_.frame_num) ? e.frame_num - max_frame_num_ : e.frame_num});
    }
    std::stable_sort(lf.begin(), lf.end(), [](const FrameRef& a, const FrameRef& b) { return a.key < b.key; });
    if (!is_b) {
      std::stable_sort(sf.begin(), sf.end(), [](const FrameRef& a, const FrameRef& b) { return a.key > b.key; });
      ref_[0] = alternate(sf, false);
      auto l = alternate(lf, true);
      ref_[0].insert(ref_[0].end(), l.begin(), l.end());
    } else {
      // refFrameList0ShortTerm: by POC; the POC of an entry for which only one field is marked is that field's; a
      // frame or complementary pair uses the POC of the fields marked (8.2.4.2.4).
      auto entry_poc = [&](const DpbEntry* e) {
        if (e->used == 3) return std::min(e->poc[0], e->poc[1]);
        return e->poc[e->used == 1 ? 0 : 1];
      };
      // when the current field is the second field of a complementary pair and the first is a reference, it is included
      std::vector<FrameRef> before, after;
      for (const FrameRef& f : sf) {
        const int p = entry_poc(f.e);
        // the "current POC" for the comparison is PicOrderCnt of the current field
        (p <= cur_poc_ ? before : after).push_back({f.e, p});
      }
      std::stable_sort(before.begin(), before.end(), [](const FrameRef& a, const FrameRef& b) { return a.key > b.key; });
      std::stable_sort(after.begin(), after.end(), [](const FrameRef& a, const FrameRef& b) { return a.key < b.key; });
      std::vector<FrameRef> l0 = before, l1 = after;
      l0.insert(l0.end(), after.begin(), after.end());
      l1.insert(l1.end(), before.begin(), before.end());
      ref_[0] = alternate(l0, false);
      ref_[1] = alternate(l1, false);
      auto l = alternate(lf, true);
      ref_[0].insert(ref_[0].end(), l.begin(), l.end());
      ref_[1].insert(ref_[1].end(), l.begin(), l.end());
    }
  }
  if (is_b && ref_[1].size() > 1 && ref_[0].size() == ref_[1].size()) {
    bool same = true;
    for (size_t i = 0; i < ref_[0].size(); ++i)
      if (ref_[0][i].id != ref_[1][i].id) same = false;
    if (same) std::swap(ref_[1][0], ref_[1][1]);
  }
  for (int l = 0; l < (is_b ? 2 : 1); ++l) {
    const size_t n = static_cast<size_t>(sh_.num_ref_idx_active[l]);
    if (ref_[l].size() > n) ref_[l].resize(n);
    // 8.2.4.3 modification
    int pred = curr_pic_num;
    size_t idx = 0;
    ref_[l].resize(n + 1);
    auto pic_num_f = [&](const RefPic& r) { return r.frame == nullptr ? INT32_MIN : (r.long_term ? max_pic_num : r.fn_wrap); };
    auto lt_pic_num_f = [&](const RefPic& r) { return r.frame == nullptr ? INT32_MIN : (r.long_term ? r.fn_wrap : INT32_MAX); };
    for (const RplMod& m : sh_.rpl_mod[l]) {
      if (idx >= n) break;
      RefPic target;
      bool found = false;
      if (m.idc == 0 || m.idc == 1) {
        const int diff = static_cast<int>(m.value) + 1;
        int no_wrap;
        if (m.idc == 0) {
          no_wrap = pred - diff;
          if (no_wrap < 0) no_wrap += max_pic_num;
        } else {
          no_wrap = pred + diff;
          if (no_wrap >= max_pic_num) no_wrap -= max_pic_num;
        }
        pred = no_wrap;
        const int pic_num = no_wrap > curr_pic_num ? no_wrap - max_pic_num : no_wrap;
        // short-term picture with PicNum == pic_num
        std::vector<RefPic>* pool = nullptr;
        (void)pool;
        for (const DpbEntry& e : pp_.dpb) {
          if (e.long_term) continue;
          if (!field_pic_) {
            if (e.used != 3) continue;
            RefPic r = make(e, 0);
            if (r.fn_wrap == pic_num) {
              target = r;
              found = true;
              break;
            }
          } else {
            for (int p = 0; p < 2 && !found; ++p) {
              if (!(e.used & (1 << p))) continue;
              RefPic r = make(e, 1 + p);
              r.fn_wrap = 2 * ((e.frame_num > pp_.frame_num) ? e.frame_num - max_frame_num_ : e.frame_num) + (p == (bottom_ ? 1 : 0) ? 1 : 0);
              if (r.fn_wrap == pic_num) {
                target = r;
                found = true;
              }
            }
            if (found) break;
          }
        }
      } else if (m.idc == 2) {
        const int lt_num = static_cast<int>(m.value);
        for (const DpbEntry& e : pp_.dpb) {
          if (!e.long_term) continue;
          if (!field_pic_) {
            if (e.used != 3) continue;
            RefPic r = make(e, 0);
            if (r.fn_wrap == lt_num) {
              target = r;
              found = true;
              break;
            }
          } else {
            for (int p = 0; p < 2 && !found; ++p) {
              if (!(e.used & (1 << p))) continue;
              RefPic r = make(e, 1 + p);
              r.fn_wrap = e.frame_num * 2 + (p == (bottom_ ? 1 : 0) ? 1 : 0);
              if (r.fn_wrap == lt_num) {
                target = r;
                found = true;
              }
            }
            if (found) break;
          }
        }
      } else {
        continue;   // MVC view index modification: not for this decoder
      }
      if (!found) return fail("a reference picture list modification names a picture that is not in the DPB");
      // insert at idx, shift the rest, drop the later duplicate
      for (size_t c = n; c > idx; --c) ref_[l][c] = ref_[l][c - 1];
      ref_[l][idx++] = target;
      size_t nidx = idx;
      for (size_t c = idx; c <= n; ++c) {
        const bool dup = target.long_term ? (lt_pic_num_f(ref_[l][c]) == target.fn_wrap) : (pic_num_f(ref_[l][c]) == target.fn_wrap && !ref_[l][c].long_term);
        if (!dup) ref_[l][nidx++] = ref_[l][c];
      }
    }
    ref_[l].resize(n);
    num_ref_[l] = static_cast<int>(n);
  }
  return true;
}

// ---- slice ------------------------------------------------------------------------------------------------
bool Dec::decode_slice(const SliceData& sd) {
  if (sd.size < 2) return fail("empty slice NAL unit");
  const int ref_idc = sd.nal[0] >> 5 & 3, type = sd.nal[0] & 31;
  if (type != 1 && type != 5) return fail("only coded slices of non-IDR and IDR pictures are decoded (no data partitioning)");
  rbsp_ = unescape(sd.nal, sd.size, 1);
  br_ = BitReader(rbsp_.data(), rbsp_.size());
  if (!parse_slice_header(br_, hc_, ref_idc, type, &sh_)) return fail("bad slice header");
  if (sh_.slice_type == kSP || sh_.slice_type == kSI) return fail("SP and SI slices are not decoded");
  if (sh_.redundant_pic_cnt > 0) return true;   // a redundant slice: the primary coded picture is decoded instead
  if (sh_.qp < -26 || sh_.qp > 51) return fail("slice QP out of range");
  cabac_ = pp_.cabac != 0;
  slice_id_ = static_cast<int>(pd_->slices.size());
  SliceParams sp;
  sp.disable_idc = sh_.disable_deblocking_filter_idc;
  sp.alpha_off = sh_.alpha_c0_offset;
  sp.beta_off = sh_.beta_offset;
  sp.is_b = sh_.is_b();
  pd_->slices.push_back(sp);
  qp_ = sh_.qp;
  if (!build_ref_lists()) return false;
  for (int l = 0; l < 2; ++l)
    for (int i = 0; i < num_ref_[l]; ++i)
      if (!ref_[l][i].frame && !sh_.is_intra()) {
        // A missing reference (a gap in frame_num, or a lost picture): conceal by using the oldest available picture.
        for (const DpbEntry& e : pp_.dpb)
          if (e.frame) {
            RefPic r;
            r.frame = e.frame;
            r.structure = field_pic_ ? 1 + (bottom_ ? 1 : 0) : 0;
            r.poc = std::min(e.poc[0], e.poc[1]);
            r.id = e.frame->uid * 4 + r.structure;
            ref_[l][i] = r;
            break;
          }
        if (!ref_[l][i].frame) return fail("reference picture missing");
      }
  int addr = sh_.first_mb * (mbaff_ ? 2 : 1);
  if (addr >= nmb_) return fail("first_mb_in_slice out of range");
  if (cabac_) {
    while (!br_.byte_aligned()) br_.bit();
    cab_.init_contexts(sh_.qp, sh_.cabac_init_idc, sh_.is_intra());
    cab_.start(&br_);
  }
  prev_mb_skipped_ = false;
  last_qp_delta_ = 0;
  const bool intra_slice = sh_.is_intra();
  bool more = true;
  while (more) {
    if (addr >= nmb_) return fail("slice runs past the end of the picture");
    if (br_.overrun()) return fail("slice data ended early");
    bool skipped = false;
    if (!cabac_) {
      if (!intra_slice) {
        const int run = static_cast<int>(br_.ue());
        for (int i = 0; i < run; ++i) {
          if (addr >= nmb_) return fail("mb_skip_run past the end of the picture");
          if (!decode_skipped_mb(addr, i == run - 1)) return false;
          ++addr;
        }
        prev_mb_skipped_ = run > 0;
        if (run > 0 && !br_.more_rbsp_data()) break;
        if (addr >= nmb_) return fail("slice runs past the end of the picture");
      }
      if (!decode_coded_mb(addr)) return false;
      prev_mb_skipped_ = false;
      ++addr;
      more = br_.more_rbsp_data();
    } else {
      if (!decode_mb_cabac(addr, &skipped, &more)) return false;
      ++addr;
    }
  }
  return true;
}

// ---- macroblock drivers --------------------------------------------------------------------------------------------

void Dec::finalize_motion() {
  for (int b8 = 0; b8 < 4; ++b8) {
    mb_->pred_flags[b8] = static_cast<uint8_t>(pred_flags_of(b8));
    for (int l = 0; l < 2; ++l) mb_->ref_id[l][b8] = mb_->ref_idx[l][b8] >= 0 ? ref_id_of(l, mb_->ref_idx[l][b8]) : -1;
  }
}

void Dec::finish_mb() {
  for (int c = 0; c < 2; ++c) {
    const int qpi = clip3(0, 51, mb_->qp + pp_.chroma_qp_offset[c]);
    mb_->qpc[c] = static_cast<int8_t>(qpc_of(qpi));
  }
}

// A macroblock whose syntax has been parsed: build its samples.
void Dec::complete_mb() {
  finish_mb();
#ifdef VGPU_H264_DEBUG
  if (std::getenv("VGPU_H264_TRACE")) {
    std::fprintf(stderr, "mb %d f%d %s kind %d t8 %d cbp %02x qp %d i16 %d chroma %d shape %d ref0 %d mv0 %d,%d ref1 %d mv1 %d,%d\n", addr_, mb_field_ ? 1 : 0, mb_->intra ? "intra" : "inter", mb_->kind, mb_->t8x8, mb_->cbp, mb_->qp, i16_mode_,
                 mb_->chroma_mode, part_shape_, mb_->ref_idx[0][0], mb_->mv[0][0][0], mb_->mv[0][0][1], mb_->ref_idx[1][0], mb_->mv[1][0][0], mb_->mv[1][0][1]);
  }
#endif
  if (mb_->intra) {
    recon_intra();
  } else {
    inter_predict_mb();
    recon_inter_residual();
  }
}

// P_Skip and B_Skip: motion inferred, no residual.
void Dec::do_skip_mb() {
#ifdef VGPU_H264_DEBUG
  if (std::getenv("VGPU_H264_TRACE")) std::fprintf(stderr, "mb %d f%d skip\n", addr_, mb_field_ ? 1 : 0);
#endif
  is_skip_ = true;
  mb_->skip = true;
  mb_->intra = false;
  mb_->kind = kInter;
  mb_->cbp = 0;
  cbp_ = 0;
  mb_->qp = static_cast<int8_t>(qp_);
  last_qp_delta_ = 0;
  blk_done_ = 0;
  if (sh_.is_b()) {
    mb_->direct16 = true;
    mb_->direct8 = 0xF;
    derive_direct(-1);
  } else {
    derive_pskip();
  }
  finalize_motion();
  finish_mb();
  inter_predict_mb();
}

bool Dec::decode_skipped_mb(int addr, bool last_in_run) {
  if (mbaff_) {
    if ((addr & 1) == 0) {
      pair_field_ = infer_pair_field(addr);
      // a skipped top macroblock whose bottom macroblock is coded takes the flag the bottom one carries: it is the next bit
      if (last_in_run && br_.more_rbsp_data()) pair_field_ = br_.peek(1) != 0;
    }
    mb_field_ = pair_field_;
  }
  setup_mb(addr);
  do_skip_mb();
  return true;
}

bool Dec::decode_coded_mb(int addr) {
  if (mbaff_) {
    if ((addr & 1) == 0 || prev_mb_skipped_) pair_field_ = br_.u(1) != 0;   // mb_field_decoding_flag
    mb_field_ = pair_field_;
  }
  setup_mb(addr);
  if (!parse_macroblock()) return false;
  complete_mb();
  return true;
}

bool Dec::decode_mb_cabac(int addr, bool* skipped, bool* more) {
  const bool top = (addr & 1) == 0;
  if (mbaff_) {
    if (top) {
      pair_field_ = infer_pair_field(addr);
      next_skip_known_ = false;
    }
    mb_field_ = pair_field_;
  }
  setup_mb(addr);
  bool skip = false;
  bool field_flag_read = false;   // the pair's mb_field_decoding_flag has been read (with the top macroblock skipped)
  if (!sh_.is_intra()) {
    if (mbaff_ && !top && next_skip_known_) skip = next_skip_;
    else skip = cabac_mb_skip() != 0;
  }
  if (skip && mbaff_ && top) {
    // The bottom macroblock's mb_skip_flag, and its mb_field_decoding_flag if it is coded, come next in the stream; the
    // top macroblock needs them: a skipped macroblock of a field pair is predicted as a field macroblock.
    mb_->skip = true;
    setup_mb(addr + 1);
    next_skip_ = cabac_mb_skip() != 0;
    next_skip_known_ = true;
    if (!next_skip_) {
      pair_field_ = cabac_field_flag() != 0;
      field_flag_read = true;
    }
    mb_field_ = pair_field_;
    setup_mb(addr);
  }
  if (skip) {
    do_skip_mb();
  } else {
    if (mbaff_ && top) {
      pair_field_ = cabac_field_flag() != 0;
      mb_field_ = pair_field_;
      setup_mb(addr);
    }
    (void)field_flag_read;
    if (!parse_macroblock()) return false;
    complete_mb();
  }
  *skipped = skip;
  prev_mb_skipped_ = skip;
  if (mbaff_ && top) {
    *more = true;
  } else {
    *more = cab_.terminate() == 0;
  }
  return !br_.overrun();
}
