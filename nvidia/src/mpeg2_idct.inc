// The inverse DCT of 7.5 / Annex A: a double precision separable transform, rounded to nearest (halves up), saturated to -256..255.
// (Reference version, to be replaced by the card's.)
// The basis: c[x][u] = (u == 0 ? sqrt(1/2) : 1) * cos((2x + 1) u pi / 16) / 2.
const double (*idct_basis())[8] {
  static const struct Table {
    double c[8][8];
    Table() {
      for (int x = 0; x < 8; ++x)
        for (int u = 0; u < 8; ++u) c[x][u] = (u == 0 ? std::sqrt(0.5) : 1.0) * std::cos((2 * x + 1) * u * 3.14159265358979323846 / 16) / 2;
    }
  } t;
  return t.c;
}

// The plain loops (the vector form below gives their bits: tests/unit/test_video_simd.cpp keeps a copy of them as its reference).
void idct_scalar(int16_t* blk) {
  const double(*c)[8] = idct_basis();
  double tmp[64];
  for (int v = 0; v < 8; ++v)
    for (int x = 0; x < 8; ++x) {
      double s = 0;
      for (int u = 0; u < 8; ++u) s += c[x][u] * blk[v * 8 + u];
      tmp[v * 8 + x] = s;
    }
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y) {
      double s = 0;
      for (int v = 0; v < 8; ++v) s += c[y][v] * tmp[v * 8 + x];
      const int r = static_cast<int>(std::floor(s + 0.5));
      blk[y * 8 + x] = static_cast<int16_t>(std::min(std::max(r, -256), 255));
    }
}

void idct(int16_t* blk) {
  if (vgpu::videosimd::avx2_available()) vgpu::videosimd::mpeg2_idct(blk, idct_basis());
  else idct_scalar(blk);
}
