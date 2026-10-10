// Part of hevc_decode.cpp: intra sample prediction (8.4.4.2).

namespace {
const int8_t kIntraPredAngle[35] = {0,  0,  32, 26, 21, 17, 13, 9,  5,   2,   0,   -2,  -5,  -9,  -13, -17, -21, -26,
                                    -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32};
const int16_t kInvAngle[35] = {0,     0,     0,    0,    0,    0,    0,    0,    0,    0,    0,    -4096, -1638, -910, -630, -482, -390, -315,
                               -256,  -315,  -390, -482, -630, -910, -1638, -4096, 0,   0,    0,    0,     0,     0,    0,    0,    0};
}  // namespace

void Dec::intra_predict(int x_cmp, int y_cmp, int log2_tb, int c_idx, int mode) {
  const int n = 1 << log2_tb;
  std::vector<uint16_t>& pl = c_idx == 0 ? F->y : (c_idx == 1 ? F->u : F->v);
  const int stride = c_idx == 0 ? F->stride_y : F->stride_c;
  const int bd = c_idx == 0 ? bd_y : bd_c;
  const int sub = c_idx == 0 ? 1 : 2;       // SubWidthC = SubHeightC at 4:2:0
  const int x_tb_y = x_cmp * sub, y_tb_y = y_cmp * sub;
  // p[-1][-1 .. 2n-1] -> left[0 .. 2n], p[0 .. 2n-1][-1] -> top[1 .. 2n] (top[0] is the corner)
  int left[129], top[129];
  bool avail_left[129], avail_top[129];
  const int unit = c_idx == 0 ? 4 : 2;   // samples sharing one availability decision
  auto usable = [&](int xn_cmp, int yn_cmp) {
    const int xn = xn_cmp * sub, yn = yn_cmp * sub;
    if (!avail_zs(x_tb_y, y_tb_y, xn, yn)) return false;
    if (P.constrained_intra_pred && pred_mode_[static_cast<size_t>(idx4(xn, yn))] != 2) return false;
    return true;
  };
  bool any = false, all = true;
  // corner
  {
    const bool a = usable(x_cmp - 1, y_cmp - 1);
    avail_left[0] = avail_top[0] = a;
    if (a) left[0] = top[0] = pl[static_cast<size_t>(y_cmp - 1) * static_cast<size_t>(stride) + static_cast<size_t>(x_cmp - 1)];
    any |= a;
    all &= a;
  }
  for (int y = 0; y < 2 * n; y += unit) {
    const bool a = usable(x_cmp - 1, y_cmp + y);
    for (int k = 0; k < unit; ++k) {
      avail_left[1 + y + k] = a;
      if (a) left[1 + y + k] = pl[static_cast<size_t>(y_cmp + y + k) * static_cast<size_t>(stride) + static_cast<size_t>(x_cmp - 1)];
    }
    any |= a;
    all &= a;
  }
  for (int x = 0; x < 2 * n; x += unit) {
    const bool a = usable(x_cmp + x, y_cmp - 1);
    for (int k = 0; k < unit; ++k) {
      avail_top[1 + x + k] = a;
      if (a) top[1 + x + k] = pl[static_cast<size_t>(y_cmp - 1) * static_cast<size_t>(stride) + static_cast<size_t>(x_cmp + x + k)];
    }
    any |= a;
    all &= a;
  }
  // 8.4.4.2.2 substitution
  if (!any) {
    for (int i = 0; i <= 2 * n; ++i) left[i] = top[i] = 1 << (bd - 1);
  } else if (!all) {
    if (!avail_left[2 * n]) {
      int v = 0;
      bool found = false;
      for (int i = 2 * n - 1; i >= 0 && !found; --i)
        if (avail_left[i]) {
          v = left[i];
          found = true;
        }
      for (int i = 1; i <= 2 * n && !found; ++i)
        if (avail_top[i]) {
          v = top[i];
          found = true;
        }
      left[2 * n] = v;
      avail_left[2 * n] = true;
    }
    for (int i = 2 * n - 1; i >= 0; --i)
      if (!avail_left[i]) left[i] = left[i + 1];
    top[0] = left[0];
    for (int i = 1; i <= 2 * n; ++i)
      if (!avail_top[i]) top[i] = top[i - 1];
  }
  top[0] = left[0];
  // 8.4.4.2.3 filtering of neighbouring samples
  if (!P.intra_smoothing_disabled && c_idx == 0) {
    bool filter = false;
    if (mode != 1 && n != 4) {
      const int min_dist = std::min(std::abs(mode - 26), std::abs(mode - 10));
      const int thres = n == 8 ? 7 : (n == 16 ? 1 : 0);
      filter = min_dist > thres;
    }
    if (filter) {
      int fl[129], ft[129];
      const bool bi_int = P.strong_intra_smoothing && c_idx == 0 && n == 32 && std::abs(left[0] + top[2 * n] - 2 * top[n]) < (1 << (bd - 5)) &&
                          std::abs(left[0] + left[2 * n] - 2 * left[n]) < (1 << (bd - 5));
      if (bi_int) {
        fl[0] = ft[0] = left[0];
        for (int i = 0; i < 63; ++i) {
          fl[1 + i] = ((63 - i) * left[0] + (i + 1) * left[64] + 32) >> 6;
          ft[1 + i] = ((63 - i) * top[0] + (i + 1) * top[64] + 32) >> 6;
        }
        fl[64] = left[64];
        ft[64] = top[64];
      } else {
        fl[0] = ft[0] = (left[1] + 2 * left[0] + top[1] + 2) >> 2;
        for (int i = 1; i < 2 * n; ++i) {
          fl[i] = (left[i + 1] + 2 * left[i] + left[i - 1] + 2) >> 2;
          ft[i] = (top[i + 1] + 2 * top[i] + top[i - 1] + 2) >> 2;
        }
        fl[2 * n] = left[2 * n];
        ft[2 * n] = top[2 * n];
        // the corner uses the unfiltered neighbours; the neighbour at index 1 of both arrays is p[-1][0] / p[0][-1]
        fl[0] = ft[0] = (left[1] + 2 * left[0] + top[1] + 2) >> 2;
        // index 1 of the filtered arrays uses the corner as its other neighbour (left[0] / top[0])
      }
      std::memcpy(left, fl, sizeof(int) * static_cast<size_t>(2 * n + 1));
      std::memcpy(top, ft, sizeof(int) * static_cast<size_t>(2 * n + 1));
    }
  }
  uint16_t* dst = &pl[static_cast<size_t>(y_cmp) * static_cast<size_t>(stride) + static_cast<size_t>(x_cmp)];
  const int maxv = (1 << bd) - 1;
  // p[-1][y] = left[1 + y], p[x][-1] = top[1 + x]
  if (mode == 0) {   // planar
    const int tr = top[1 + n], bl = left[1 + n];
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) dst[y * stride + x] = static_cast<uint16_t>(((n - 1 - x) * left[1 + y] + (x + 1) * tr + (n - 1 - y) * top[1 + x] + (y + 1) * bl + n) >> (log2_tb + 1));
  } else if (mode == 1) {   // DC
    int sum = n;
    for (int i = 0; i < n; ++i) sum += top[1 + i] + left[1 + i];
    const int dc = sum >> (log2_tb + 1);
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) dst[y * stride + x] = static_cast<uint16_t>(dc);
    if (c_idx == 0 && n < 32) {
      dst[0] = static_cast<uint16_t>((left[1] + 2 * dc + top[1] + 2) >> 2);
      for (int x = 1; x < n; ++x) dst[x] = static_cast<uint16_t>((top[1 + x] + 3 * dc + 2) >> 2);
      for (int y = 1; y < n; ++y) dst[y * stride] = static_cast<uint16_t>((left[1 + y] + 3 * dc + 2) >> 2);
    }
  } else {   // angular
    const int angle = kIntraPredAngle[mode];
    const bool disable_boundary = P.implicit_rdpcm && cu_transquant_bypass_;
    int ref_buf[3 * 64 + 8];
    int* ref = ref_buf + 64 + 4;
    if (mode >= 18) {
      for (int x = 0; x <= n; ++x) ref[x] = top[x];   // ref[x] = p[-1 + x][-1]
      if (angle < 0) {
        const int last = (n * angle) >> 5;
        if (last < -1) {
          const int inv = kInvAngle[mode];
          for (int x = -1; x >= last; --x) ref[x] = left[(((x * inv + 128) >> 8))];   // p[-1][-1 + ((x * inv + 128) >> 8)] = left[(x*inv+128)>>8]
        }
      } else {
        for (int x = n + 1; x <= 2 * n; ++x) ref[x] = top[x];
      }
      for (int y = 0; y < n; ++y) {
        const int idx = ((y + 1) * angle) >> 5, fact = ((y + 1) * angle) & 31;
        for (int x = 0; x < n; ++x) {
          if (fact) dst[y * stride + x] = static_cast<uint16_t>(((32 - fact) * ref[x + idx + 1] + fact * ref[x + idx + 2] + 16) >> 5);
          else dst[y * stride + x] = static_cast<uint16_t>(ref[x + idx + 1]);
        }
      }
      if (mode == 26 && c_idx == 0 && n < 32 && !disable_boundary)
        for (int y = 0; y < n; ++y) dst[y * stride] = static_cast<uint16_t>(std::max(0, std::min(maxv, top[1] + ((left[1 + y] - left[0]) >> 1))));
    } else {
      for (int x = 0; x <= n; ++x) ref[x] = left[x];   // ref[x] = p[-1][-1 + x]
      if (angle < 0) {
        const int last = (n * angle) >> 5;
        if (last < -1) {
          const int inv = kInvAngle[mode];
          for (int x = -1; x >= last; --x) ref[x] = top[(((x * inv + 128) >> 8))];
        }
      } else {
        for (int x = n + 1; x <= 2 * n; ++x) ref[x] = left[x];
      }
      for (int x = 0; x < n; ++x) {
        const int idx = ((x + 1) * angle) >> 5, fact = ((x + 1) * angle) & 31;
        for (int y = 0; y < n; ++y) {
          if (fact) dst[y * stride + x] = static_cast<uint16_t>(((32 - fact) * ref[y + idx + 1] + fact * ref[y + idx + 2] + 16) >> 5);
          else dst[y * stride + x] = static_cast<uint16_t>(ref[y + idx + 1]);
        }
      }
      if (mode == 10 && c_idx == 0 && n < 32 && !disable_boundary)
        for (int x = 0; x < n; ++x) dst[x] = static_cast<uint16_t>(std::max(0, std::min(maxv, left[1] + ((top[1 + x] - top[0]) >> 1))));
    }
  }
}
