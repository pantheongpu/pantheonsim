// The inverse DCT of 7.5 / Annex A: a double precision separable transform, rounded to nearest (halves up), saturated to -256..255.
// (Reference version, to be replaced by the card's.)
void idct(int16_t* blk) {
  static double c[8][8];
  static bool init = false;
  if (!init) {
    for (int x = 0; x < 8; ++x)
      for (int u = 0; u < 8; ++u) c[x][u] = (u == 0 ? std::sqrt(0.5) : 1.0) * std::cos((2 * x + 1) * u * 3.14159265358979323846 / 16) / 2;
    init = true;
  }
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
