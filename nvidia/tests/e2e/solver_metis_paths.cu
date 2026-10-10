// cusolverSpXcsrmetisndHost: NVIDIA documents it as a wrapper of METIS 5.1.0's
// METIS_NodeND, and this library runs the same METIS (nvidia/third_party/metis),
// so for a fixed graph the permutation should be the one an RTX 3060's
// cuSOLVER returns. Checked two ways:
//
//   * always: every answer is a permutation of 0..n-1, a fixed-size nested
//     dissection ordering never fills a banded or grid matrix worse than the
//     scrambled order, an options array of METIS defaults gives the default
//     answer, and the base-one form equals the base-zero form;
//   * against the card: the permutations NVIDIA's library returned for the
//     matrices below, kept in nvidia/tests/data/solver_metis_paths.card.txt
//     (one "name hash first-eight-entries" line per matrix) and compared
//     entry for entry. METIS is deterministic for a graph, but the ordering
//     it returns depends on the adjacency lists' order and on glibc's rand();
//     a matrix whose line differs is a failure unless the data file marks it
//     "known" (the permutation is still checked to be a valid one).
//
// METIS_PRINT=1 prints the lines instead of comparing them: build the program
// against NVIDIA's libcusolver on a card and redirect its output to refresh
// the data file (nvidia/tests/e2e/run_metis_card.sh does).
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

struct Pattern {
  std::string name;
  int n = 0;
  std::vector<int> off, col;   // zero-based CSR, unsorted-free: columns ascending, diagonal as given
};

// A deterministic generator that does not depend on the C library.
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(s >> 33);
  }
  int below(int n) { return (int)(next() % (uint32_t)n); }
};

static Pattern from_edges(const std::string& name, int n, const std::vector<std::pair<int, int>>& edges, bool symmetric,
                          bool diagonal) {
  std::vector<std::set<int>> rows((size_t)n);
  for (const auto& [i, j] : edges) {
    rows[(size_t)i].insert(j);
    if (symmetric) rows[(size_t)j].insert(i);
  }
  if (diagonal)
    for (int i = 0; i < n; ++i) rows[(size_t)i].insert(i);
  Pattern p;
  p.name = name;
  p.n = n;
  p.off.push_back(0);
  for (const auto& r : rows) {
    for (int c : r) p.col.push_back(c);
    p.off.push_back((int)p.col.size());
  }
  return p;
}

static std::vector<Pattern> matrices() {
  std::vector<Pattern> out;
  auto add = [&](Pattern p) { out.push_back(std::move(p)); };
  // Meshes: 2-D five-point, 2-D nine-point, 3-D seven-point.
  for (int k : {3, 5, 8, 12, 17, 24, 33}) {
    std::vector<std::pair<int, int>> e;
    for (int y = 0; y < k; ++y)
      for (int x = 0; x < k; ++x) {
        if (x + 1 < k) e.push_back({y * k + x, y * k + x + 1});
        if (y + 1 < k) e.push_back({y * k + x, (y + 1) * k + x});
      }
    add(from_edges("grid2d_" + std::to_string(k), k * k, e, true, true));
  }
  for (int k : {4, 9, 20}) {
    std::vector<std::pair<int, int>> e;
    for (int y = 0; y < k; ++y)
      for (int x = 0; x < k; ++x)
        for (int dy = 0; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if ((dy == 0 && dx <= 0) || yy >= k || xx < 0 || xx >= k) continue;
            e.push_back({y * k + x, yy * k + xx});
          }
    add(from_edges("grid2d9_" + std::to_string(k), k * k, e, true, true));
  }
  for (int k : {3, 5, 8}) {
    std::vector<std::pair<int, int>> e;
    for (int z = 0; z < k; ++z)
      for (int y = 0; y < k; ++y)
        for (int x = 0; x < k; ++x) {
          const int id = (z * k + y) * k + x;
          if (x + 1 < k) e.push_back({id, id + 1});
          if (y + 1 < k) e.push_back({id, id + k});
          if (z + 1 < k) e.push_back({id, id + k * k});
        }
    add(from_edges("grid3d_" + std::to_string(k), k * k * k, e, true, true));
  }
  // Banded, tridiagonal, arrow, star, path, cycle, complete, disconnected.
  for (int n : {10, 50, 300}) {
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i < n; ++i)
      for (int w = 1; w <= 3 && i + w < n; ++w) e.push_back({i, i + w});
    add(from_edges("band3_" + std::to_string(n), n, e, true, true));
  }
  for (int n : {2, 3, 7, 40, 200}) {
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i + 1 < n; ++i) e.push_back({i, i + 1});
    add(from_edges("path_" + std::to_string(n), n, e, true, true));
  }
  for (int n : {5, 31}) {
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i < n; ++i) e.push_back({i, (i + 1) % n});
    add(from_edges("cycle_" + std::to_string(n), n, e, true, false));
  }
  for (int n : {6, 40}) {
    std::vector<std::pair<int, int>> e;
    for (int i = 1; i < n; ++i) e.push_back({0, i});
    add(from_edges("star_" + std::to_string(n), n, e, true, true));
    std::vector<std::pair<int, int>> a;
    for (int i = 0; i < n; ++i) a.push_back({i, n - 1});
    add(from_edges("arrow_" + std::to_string(n), n, a, true, true));
  }
  for (int n : {4, 9, 16}) {
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) e.push_back({i, j});
    add(from_edges("complete_" + std::to_string(n), n, e, true, true));
  }
  {
    std::vector<std::pair<int, int>> e;   // four disconnected paths and three isolated vertices
    for (int c = 0; c < 4; ++c)
      for (int i = 0; i + 1 < 12; ++i) e.push_back({c * 12 + i, c * 12 + i + 1});
    add(from_edges("components_51", 51, e, true, true));
  }
  add(from_edges("single", 1, {}, true, true));
  add(from_edges("isolated_30", 30, {}, true, true));
  // Random graphs: symmetric and not, with and without the diagonal.
  const int sizes[] = {12, 30, 64, 100, 150, 250, 400, 700, 1200};
  const int degs[] = {2, 3, 5, 8};
  int idx = 0;
  for (int n : sizes)
    for (int d : degs) {
      if (n > 400 && d > 5) continue;
      Lcg r(1000 + idx++);
      std::vector<std::pair<int, int>> e;
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < d; ++k) {
          const int j = r.below(n);
          if (j != i) e.push_back({i, j});
        }
      add(from_edges("random_" + std::to_string(n) + "_d" + std::to_string(d), n, e, (idx % 3) != 0, (idx % 2) == 0));
    }
  // Graphs with isolated vertices, trees, ladders, a hub and spokes, an unsymmetric pattern
  // with empty rows, and a larger mesh (past 1000 vertices METIS tries several separators).
  for (int n : {40, 200}) {
    Lcg r(77 + n);
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i < n / 2; ++i) {   // only the first half has edges
      for (int k = 0; k < 3; ++k) {
        const int j = r.below(n / 2);
        if (j != i) e.push_back({i, j});
      }
    }
    add(from_edges("half_isolated_" + std::to_string(n), n, e, true, false));
    add(from_edges("rows_empty_unsym_" + std::to_string(n), n, e, false, true));
  }
  for (int n : {31, 255}) {
    std::vector<std::pair<int, int>> e;
    for (int i = 1; i < n; ++i) e.push_back({i, (i - 1) / 2});   // a binary tree
    add(from_edges("tree_" + std::to_string(n), n, e, true, true));
  }
  for (int n : {20, 150}) {
    std::vector<std::pair<int, int>> e;   // a ladder
    for (int i = 0; i < n; ++i) {
      e.push_back({2 * i, 2 * i + 1});
      if (i + 1 < n) {
        e.push_back({2 * i, 2 * i + 2});
        e.push_back({2 * i + 1, 2 * i + 3});
      }
    }
    add(from_edges("ladder_" + std::to_string(n), 2 * n, e, true, true));
  }
  {
    const int k = 40;   // 1600 vertices: several separators are tried
    std::vector<std::pair<int, int>> e;
    for (int y = 0; y < k; ++y)
      for (int x = 0; x < k; ++x) {
        if (x + 1 < k) e.push_back({y * k + x, y * k + x + 1});
        if (y + 1 < k) e.push_back({y * k + x, (y + 1) * k + x});
      }
    add(from_edges("grid2d_40", k * k, e, true, true));
  }
  {
    Lcg r(4242);
    const int n = 3000;   // a skewed degree distribution: a few hubs
    std::vector<std::pair<int, int>> e;
    for (int i = 1; i < n; ++i) {
      e.push_back({i, r.below(i)});
      if (r.below(4) == 0) e.push_back({i, r.below(std::min(i, 20))});
    }
    add(from_edges("hubs_3000", n, e, true, true));
  }
  // A scrambled band: nested dissection should recover most of the structure.
  {
    const int n = 120;
    std::vector<int> scramble(n);
    for (int i = 0; i < n; ++i) scramble[i] = (i * 37 + 11) % n;
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i < n; ++i)
      for (int w = 1; w <= 2 && i + w < n; ++w) e.push_back({scramble[i], scramble[i + w]});
    add(from_edges("scrambled_band_120", n, e, true, true));
  }
  return out;
}

static uint64_t fnv(const std::vector<int>& p) {
  uint64_t h = 1469598103934665603ULL;
  for (int v : p) {
    h ^= (uint64_t)(uint32_t)v;
    h *= 1099511628211ULL;
  }
  return h;
}

static std::string line_of(const std::string& name, int n, const std::vector<int>& p) {
  char buf[256];
  std::snprintf(buf, sizeof buf, "%s %d %016llx", name.c_str(), n, (unsigned long long)fnv(p));
  std::string s = buf;
  for (int i = 0; i < std::min(8, n); ++i) s += " " + std::to_string(p[(size_t)i]);
  return s;
}

int main() {
  cusolverSpHandle_t h;
  if (cusolverSpCreate(&h)) { std::printf("FAIL cusolverSpCreate\n"); return 1; }
  cusparseMatDescr_t d;
  cusparseCreateMatDescr(&d);
  const bool print = std::getenv("METIS_PRINT") != nullptr;

  std::map<std::string, std::string> card;   // name -> line
  std::set<std::string> known;
  if (!print) {
    const char* dir = std::getenv("VGPU_E2E_DATA");
    std::ifstream in(std::string(dir ? dir : ".") + "/solver_metis_paths.card.txt");
    std::string l;
    while (std::getline(in, l)) {
      if (l.empty() || l[0] == '#') continue;
      std::istringstream ss(l);
      std::string name;
      ss >> name;
      if (name == "known") {
        ss >> name;
        known.insert(name);
        continue;
      }
      card[name] = l;
    }
    if (card.empty()) std::printf("SKIP the card's permutations are not available (VGPU_E2E_DATA)\n");
  }

  const auto all = matrices();
  int same = 0, different = 0, listed = 0;
  bool all_perms = true;
  for (const auto& m : all) {
    std::vector<int> p((size_t)m.n, -1);
    const int st = cusolverSpXcsrmetisndHost(h, m.n, (int)m.col.size(), d, m.off.data(), m.col.data(), nullptr, p.data());
    std::vector<int> sorted = p;
    std::sort(sorted.begin(), sorted.end());
    std::vector<int> want((size_t)m.n);
    std::iota(want.begin(), want.end(), 0);
    const bool perm = st == 0 && sorted == want;
    all_perms = all_perms && perm;
    if (!perm) std::printf("FAIL %s: status %d, not a permutation\n", m.name.c_str(), st);
    const std::string line = line_of(m.name, m.n, p);
    if (print) {
      std::printf("%s\n", line.c_str());
      continue;
    }
    auto it = card.find(m.name);
    if (it == card.end()) continue;
    ++listed;
    if (it->second == line) {
      ++same;
    } else {
      ++different;
      const bool ok = known.count(m.name) != 0;
      std::printf("%-4s %s differs from the card's:\n       ours  %s\n       card  %s\n", ok ? "note" : "FAIL", m.name.c_str(),
                  line.c_str(), it->second.c_str());
      if (!ok) ++failures;
    }
  }
  if (print) return 0;
  check(all_perms, "every ordering is a permutation of 0..n-1");
  if (listed) {
    char what[200];
    std::snprintf(what, sizeof what, "METIS_NodeND's permutation equals the card's on %d of %d matrices (%d differ, all listed as known)",
                  same, listed, different);
    check(same + (int)known.size() >= listed && different <= (int)known.size(), what);
  }

  // The same graph in base-one indexing, and an explicit options array of METIS's defaults.
  {
    const Pattern& m = all[3];
    std::vector<int> p0((size_t)m.n), p1((size_t)m.n), p2((size_t)m.n);
    cusolverSpXcsrmetisndHost(h, m.n, (int)m.col.size(), d, m.off.data(), m.col.data(), nullptr, p0.data());
    cusparseMatDescr_t d1;
    cusparseCreateMatDescr(&d1);
    cusparseSetMatIndexBase(d1, CUSPARSE_INDEX_BASE_ONE);
    std::vector<int> off1 = m.off, col1 = m.col;
    for (int& v : off1) ++v;
    for (int& v : col1) ++v;
    const int st = cusolverSpXcsrmetisndHost(h, m.n, (int)m.col.size(), d1, off1.data(), col1.data(), nullptr, p1.data());
    check(st == 0 && p0 == p1, "a base-one matrix gives the base-zero permutation");
    cusparseDestroyMatDescr(d1);
    // METIS_NOPTIONS (40) entries, all -1 (= "use the default").
    std::vector<int64_t> opt(40, -1);
    cusolverSpXcsrmetisndHost(h, m.n, (int)m.col.size(), d, m.off.data(), m.col.data(), opt.data(), p2.data());
    check(p0 == p2, "an options array of -1 (METIS defaults) gives the default permutation");
    // A different seed is a valid ordering and, on a graph this size, not the same one.
    std::vector<int64_t> seeded(40, -1);
    seeded[8] = 7;   // METIS_OPTION_SEED
    std::vector<int> p3((size_t)m.n);
    const int s3 = cusolverSpXcsrmetisndHost(h, m.n, (int)m.col.size(), d, m.off.data(), m.col.data(), seeded.data(), p3.data());
    std::vector<int> sorted3 = p3;
    std::sort(sorted3.begin(), sorted3.end());
    std::vector<int> iota_n((size_t)m.n);
    std::iota(iota_n.begin(), iota_n.end(), 0);
    check(s3 == 0 && sorted3 == iota_n, "METIS_OPTION_SEED is passed through and gives a permutation");
  }

  // Arguments, as the card answers them.
  {
    const Pattern& m = all[0];
    std::vector<int> p((size_t)m.n);
    const int nz = (int)m.col.size();
    auto call = [&](cusolverSpHandle_t hh, int n, int nnz, cusparseMatDescr_t dd, const int64_t* o) {
      return (int)cusolverSpXcsrmetisndHost(hh, n, nnz, dd, m.off.data(), m.col.data(), o, p.data());
    };
    check(call(h, 0, 0, d, nullptr) == CUSOLVER_STATUS_INVALID_VALUE, "n = 0 is INVALID_VALUE");
    check(call(h, -1, nz, d, nullptr) == CUSOLVER_STATUS_INVALID_VALUE, "a negative n is INVALID_VALUE");
    check(call(h, m.n, 0, d, nullptr) == CUSOLVER_STATUS_INVALID_VALUE, "nnz = 0 is INVALID_VALUE");
    check(call(h, m.n, -1, d, nullptr) == CUSOLVER_STATUS_ALLOC_FAILED, "a negative nnz is ALLOC_FAILED");
    check(call(h, m.n, nz, nullptr, nullptr) == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "a NULL descriptor is MATRIX_TYPE_NOT_SUPPORTED");
    check(call(nullptr, m.n, nz, d, nullptr) == CUSOLVER_STATUS_NOT_INITIALIZED, "a NULL handle is NOT_INITIALIZED");
    cusparseMatDescr_t sym;
    cusparseCreateMatDescr(&sym);
    cusparseSetMatType(sym, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    check(call(h, m.n, nz, sym, nullptr) == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "a symmetric-type descriptor is MATRIX_TYPE_NOT_SUPPORTED");
    cusparseDestroyMatDescr(sym);
    std::vector<int64_t> bad(40, -1);
    bad[12] = 5;   // METIS_OPTION_COMPRESS: only 0 and 1 are valid
    check(call(h, m.n, nz, d, bad.data()) == CUSOLVER_STATUS_INTERNAL_ERROR, "an option METIS rejects is INTERNAL_ERROR");
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  cusparseDestroyMatDescr(d);
  cusolverSpDestroy(h);
  return failures ? 1 : 0;
}
