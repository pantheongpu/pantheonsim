// Holds amd/src/mfma_models.hpp to the vectors tools/mfma-models/model_ref.py writes, one file, nothing else linked:
//   g++ -std=c++20 -O2 -I amd/src tools/mfma-models/mfma_models_check.cpp -o mfma_models_check
//   tools/mfma-models/model_ref.py gen v.txt --n 20000 --seed 7 && ./mfma_models_check v.txt
// Prints the mismatches (at most 10) and a count per group; the exit status is 1 if there is one.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mfma_models.hpp"

namespace kh = vgpu::amd::khattak;

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static double decode(char t, uint32_t c) {
  switch (t) {
    case 'h': { const uint16_t u = static_cast<uint16_t>(c); _Float16 h; std::memcpy(&h, &u, 2); return static_cast<double>(h); }
    case 'b': return from_bits(c << 16);
    case 'F': return from_bits((c >> 18 & 1) << 31 | (c >> 10 & 255) << 23 | (c & 1023) << 13);
    case 'e': { const double s = c & 0x80 ? -1 : 1; const uint32_t e = c >> 3 & 15, m = c & 7; return s * (e == 0 ? m / 8.0 * 0x1p-7 : (1 + m / 8.0) * std::ldexp(1.0, int(e) - 8)); }
    default: { const double s = c & 0x80 ? -1 : 1; const uint32_t e = c >> 2 & 31, m = c & 3; return s * (e == 0 ? m / 4.0 * 0x1p-15 : (1 + m / 4.0) * std::ldexp(1.0, int(e) - 16)); }
  }
}
static std::vector<uint32_t> hex_list(const std::string& s) {
  std::vector<uint32_t> v;
  std::istringstream in(s);
  std::string w;
  while (in >> w) v.push_back(static_cast<uint32_t>(std::stoul(w, nullptr, 16)));
  return v;
}

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: mfma_models_check <vectors>\n"); return 2; }
  std::ifstream in(argv[1]);
  std::string line;
  std::map<std::string, std::pair<unsigned long, unsigned long>> per;   // group: vectors, mismatches
  int shown = 0;
  unsigned long bad = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> part;
    std::istringstream ls(line);
    std::string p;
    while (std::getline(ls, p, '|')) part.push_back(p);
    std::istringstream head(part[0]);
    std::string model, ta, tb;
    unsigned k;
    head >> model >> ta >> tb >> k;
    const auto a = hex_list(part[1]), b = hex_list(part[2]);
    std::vector<double> da, db;
    for (unsigned i = 0; i < k; ++i) da.push_back(decode(ta[0], a[i])), db.push_back(decode(tb[0], b[i]));
    const kh::Model m = model == "cdna1" ? kh::Model::Cdna1 : model == "cdna2" ? kh::Model::Cdna2 : kh::Model::Cdna3;
    const uint32_t c = hex_list(part[3])[0], want = hex_list(part[4])[0];
    const uint32_t got = bits(static_cast<float>(kh::dot(m, ta[0], tb[0], da.data(), db.data(), 1, k, from_bits(c))));
    const bool same = got == want || (from_bits(got) != from_bits(got) && from_bits(want) != from_bits(want));
    auto& g = per[model + " " + ta + tb + " K=" + std::to_string(k)];
    ++g.first;
    if (!same) {
      ++g.second, ++bad;
      if (shown++ < 10) std::printf("MISMATCH %s: got %08x, reference %08x\n  %s\n", (model + " " + ta + tb).c_str(), got, want, line.c_str());
    }
  }
  for (const auto& [name, st] : per) std::printf("%-18s vectors %lu, mismatches %lu\n", name.c_str(), st.first, st.second);
  return bad ? 1 : 0;
}
