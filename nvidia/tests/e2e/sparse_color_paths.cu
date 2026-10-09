// cusparse<t>csrcolor, checked by the properties its documentation gives and
// by what an RTX 3060's cuSPARSE (12.6.3, CUDA 13.0) shows of the rest. The
// colors themselves are not compared: NVIDIA's algorithm is a randomized
// multi-hash one that is not public, and its numbers are not even contiguous
// (a 6-node path gets colors 0, 8, 4, 8, 0, 10 and ncolors 11), so a different
// proper coloring is a correct answer. Every check here passes on the card.
//
//   * neighbours never share a color, whatever the fraction;
//   * ncolors is the largest color plus one;
//   * reordering lists the nodes by color, each color's nodes ascending;
//   * a fraction below 1 colors only part of the graph in rounds and gives
//     each remaining node a color of its own: far more colors than fraction 1;
//   * the arrays are in the matrix descriptor's index base;
//   * the fraction is read from the device in device pointer mode, and
//     ncolors is written there;
//   * a fraction outside [0, 1] is INVALID_VALUE, a symmetric matrix type
//     MATRIX_TYPE_NOT_SUPPORTED, m = 0 succeeds and writes nothing.
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

struct Graph {
  int n = 0;
  std::vector<int> off, col;
  std::vector<std::pair<int, int>> edges;
};

static Graph make(int n, const std::vector<std::pair<int, int>>& edges) {
  Graph g;
  g.n = n;
  g.edges = edges;
  std::vector<std::set<int>> rows((size_t)n);
  for (auto [i, j] : edges) {
    rows[(size_t)i].insert(j);
    rows[(size_t)j].insert(i);
  }
  for (int i = 0; i < n; ++i) rows[(size_t)i].insert(i);   // a diagonal, as matrices have
  g.off.push_back(0);
  for (auto& r : rows) {
    for (int c : r) g.col.push_back(c);
    g.off.push_back((int)g.col.size());
  }
  return g;
}

static Graph random_graph(int n, int deg, unsigned seed) {
  unsigned s = seed;
  auto next = [&] { s = s * 1664525u + 1013904223u; return (s >> 8); };
  std::vector<std::pair<int, int>> e;
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < deg; ++d) {
      const int j = (int)(next() % (unsigned)n);
      if (j != i) e.push_back({i, j});
    }
  return make(n, e);
}

struct Result {
  int status = -1;
  int ncolors = -1;
  std::vector<int> color, order;
};

static Result color(cusparseHandle_t h, const Graph& g, float fraction, cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO) {
  const int b = base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  std::vector<int> off = g.off, col = g.col;
  for (int& v : off) v += b;
  for (int& v : col) v += b;
  std::vector<float> val(g.col.size(), 1.f);
  int *dOff, *dCol, *dColoring, *dOrder;
  float* dVal;
  cudaMalloc(&dOff, off.size() * 4);
  cudaMalloc(&dCol, std::max<size_t>(1, col.size()) * 4);
  cudaMalloc(&dVal, std::max<size_t>(1, val.size()) * 4);
  cudaMalloc(&dColoring, std::max(1, g.n) * 4);
  cudaMalloc(&dOrder, std::max(1, g.n) * 4);
  cudaMemcpy(dOff, off.data(), off.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dCol, col.data(), col.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dVal, val.data(), val.size() * 4, cudaMemcpyHostToDevice);
  cusparseMatDescr_t d;
  cusparseCreateMatDescr(&d);
  cusparseSetMatIndexBase(d, base);
  cusparseColorInfo_t info;
  cusparseCreateColorInfo(&info);
  Result r;
  r.status = (int)cusparseScsrcolor(h, g.n, (int)g.col.size(), d, dVal, dOff, dCol, &fraction, &r.ncolors, dColoring, dOrder, info);
  r.color.assign((size_t)g.n, -1);
  r.order.assign((size_t)g.n, -1);
  cudaMemcpy(r.color.data(), dColoring, (size_t)g.n * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(r.order.data(), dOrder, (size_t)g.n * 4, cudaMemcpyDeviceToHost);
  cusparseDestroyColorInfo(info);
  cusparseDestroyMatDescr(d);
  cudaFree(dOff);
  cudaFree(dCol);
  cudaFree(dVal);
  cudaFree(dColoring);
  cudaFree(dOrder);
  return r;
}

static bool proper(const Graph& g, const std::vector<int>& c) {
  for (auto [i, j] : g.edges)
    if (c[(size_t)i] == c[(size_t)j]) return false;
  return true;
}

// ncolors = largest + 1; reordering sorted by color, ties ascending.
static bool consistent(const Graph& g, const Result& r, int base) {
  if (g.n == 0) return true;
  const int mx = *std::max_element(r.color.begin(), r.color.end()) - base;
  if (r.ncolors != mx + 1) return false;
  std::vector<int> seen((size_t)g.n, 0);
  for (int i = 0; i < g.n; ++i) {
    const int node = r.order[(size_t)i] - base;
    if (node < 0 || node >= g.n || seen[(size_t)node]++) return false;
    if (i > 0) {
      const int prev = r.order[(size_t)i - 1] - base;
      const int cp = r.color[(size_t)prev], cn = r.color[(size_t)node];
      if (cn < cp || (cn == cp && node < prev)) return false;
    }
  }
  return true;
}

int main() {
  cusparseHandle_t h;
  cusparseCreate(&h);

  std::vector<std::pair<int, int>> path6, grid, k4, cycle5;
  for (int i = 0; i + 1 < 6; ++i) path6.push_back({i, i + 1});
  for (int y = 0; y < 12; ++y)
    for (int x = 0; x < 12; ++x) {
      if (x + 1 < 12) grid.push_back({y * 12 + x, y * 12 + x + 1});
      if (y + 1 < 12) grid.push_back({y * 12 + x, (y + 1) * 12 + x});
    }
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j) k4.push_back({i, j});
  for (int i = 0; i < 5; ++i) cycle5.push_back({i, (i + 1) % 5});
  const std::vector<std::pair<std::string, Graph>> graphs = {
      {"path of 6", make(6, path6)},       {"12x12 grid", make(144, grid)},   {"K4", make(4, k4)},
      {"5-cycle", make(5, cycle5)},        {"5 isolated nodes", make(5, {})}, {"one node", make(1, {})},
      {"random 500 x 4", random_graph(500, 4, 1)}, {"random 2000 x 3", random_graph(2000, 3, 2)},
      {"random 3000 x 6", random_graph(3000, 6, 3)}};
  for (const auto& [name, g] : graphs)
    for (float f : {1.0f, 0.8f, 0.0f}) {
      const Result r = color(h, g, f);
      char what[160];
      std::snprintf(what, sizeof what, "%s, fraction %.1f: proper coloring, ncolors = largest + 1, reordering by color", name.c_str(), f);
      check(r.status == 0 && proper(g, r.color) && consistent(g, r, 0), what);
    }
  {
    const Graph& g = graphs[7].second;
    const Result full = color(h, g, 1.0f), part = color(h, g, 0.0f);
    char what[160];
    std::snprintf(what, sizeof what, "fraction 0 stops after a round and numbers the rest apart: %d colors against %d", part.ncolors, full.ncolors);
    check(part.ncolors > 2 * full.ncolors, what);
    std::set<int> distinct(full.color.begin(), full.color.end());
    check((int)distinct.size() <= full.ncolors, "fraction 1 uses no more distinct colors than ncolors says");
  }
  {
    const Graph& g = graphs[0].second;
    const Result z = color(h, g, 1.0f, CUSPARSE_INDEX_BASE_ZERO), o = color(h, g, 1.0f, CUSPARSE_INDEX_BASE_ONE);
    bool shifted = o.status == 0 && o.ncolors == z.ncolors;
    for (int i = 0; i < g.n; ++i) shifted = shifted && o.color[(size_t)i] == z.color[(size_t)i] + 1 && o.order[(size_t)i] == z.order[(size_t)i] + 1;
    check(shifted, "with a base-one descriptor, coloring and reordering are base-one: the base-zero answer plus 1");
    check(consistent(g, o, 1), "the base-one answer is consistent");
  }
  {
    const Graph& g = graphs[1].second;
    cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_DEVICE);
    int *dOff, *dCol, *dColoring, *dOrder, *dN;
    float *dVal, *dFrac;
    cudaMalloc(&dOff, g.off.size() * 4);
    cudaMalloc(&dCol, g.col.size() * 4);
    cudaMalloc(&dVal, g.col.size() * 4);
    cudaMalloc(&dColoring, (size_t)g.n * 4);
    cudaMalloc(&dOrder, (size_t)g.n * 4);
    cudaMalloc(&dN, 4);
    cudaMalloc(&dFrac, 4);
    const std::vector<float> val(g.col.size(), 1.f);
    const float one = 1.f;
    cudaMemcpy(dOff, g.off.data(), g.off.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dCol, g.col.data(), g.col.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dVal, val.data(), val.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dFrac, &one, 4, cudaMemcpyHostToDevice);
    cusparseMatDescr_t d;
    cusparseCreateMatDescr(&d);
    cusparseColorInfo_t info;
    cusparseCreateColorInfo(&info);
    const int st = (int)cusparseScsrcolor(h, g.n, (int)g.col.size(), d, dVal, dOff, dCol, dFrac, dN, dColoring, dOrder, info);
    int n = -1;
    cudaMemcpy(&n, dN, 4, cudaMemcpyDeviceToHost);
    std::vector<int> c((size_t)g.n);
    cudaMemcpy(c.data(), dColoring, (size_t)g.n * 4, cudaMemcpyDeviceToHost);
    check(st == 0 && n == *std::max_element(c.begin(), c.end()) + 1 && proper(g, c), "device pointer mode: fraction read from the device, ncolors written there");
    cusparseDestroyColorInfo(info);
    cusparseDestroyMatDescr(d);
    cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST);
    cudaFree(dOff); cudaFree(dCol); cudaFree(dVal); cudaFree(dColoring); cudaFree(dOrder); cudaFree(dN); cudaFree(dFrac);
  }
  {
    const Graph& g = graphs[0].second;
    check(color(h, g, -1.0f).status == CUSPARSE_STATUS_INVALID_VALUE, "a fraction below 0 is INVALID_VALUE");
    check(color(h, g, 2.0f).status == CUSPARSE_STATUS_INVALID_VALUE, "a fraction above 1 is INVALID_VALUE");
    Graph empty;
    const Result r = color(h, empty, 1.0f);
    check(r.status == 0 && r.ncolors == -1, "m = 0 succeeds and writes nothing");
    // A symmetric matrix type.
    int off[3] = {0, 1, 2}, col[2] = {0, 1};
    float v[2] = {1, 1};
    int *dO, *dC, *dCol, *dRe;
    float* dV;
    cudaMalloc(&dO, 12); cudaMalloc(&dC, 8); cudaMalloc(&dV, 8); cudaMalloc(&dCol, 8); cudaMalloc(&dRe, 8);
    cudaMemcpy(dO, off, 12, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, col, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, v, 8, cudaMemcpyHostToDevice);
    cusparseMatDescr_t d;
    cusparseCreateMatDescr(&d);
    cusparseSetMatType(d, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseColorInfo_t info;
    cusparseCreateColorInfo(&info);
    float f = 1.f;
    int nc = 0;
    check(cusparseScsrcolor(h, 2, 2, d, dV, dO, dC, &f, &nc, dCol, dRe, info) == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
          "a symmetric matrix type is MATRIX_TYPE_NOT_SUPPORTED");
    cusparseDestroyColorInfo(info);
    cusparseDestroyMatDescr(d);
    cudaFree(dO); cudaFree(dC); cudaFree(dV); cudaFree(dCol); cudaFree(dRe);
  }
  {
    // The double and complex entry points take their fraction in their own precision.
    const Graph& g = graphs[1].second;
    std::vector<double> vd(g.col.size(), 1.0);
    std::vector<cuDoubleComplex> vz(g.col.size(), make_cuDoubleComplex(1, 0));
    std::vector<cuComplex> vc(g.col.size(), make_cuComplex(1, 0));
    int *dOff, *dCol, *dColoring, *dOrder;
    void *dV;
    cudaMalloc(&dOff, g.off.size() * 4);
    cudaMalloc(&dCol, g.col.size() * 4);
    cudaMalloc(&dV, g.col.size() * 16);
    cudaMalloc(&dColoring, (size_t)g.n * 4);
    cudaMalloc(&dOrder, (size_t)g.n * 4);
    cudaMemcpy(dOff, g.off.data(), g.off.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dCol, g.col.data(), g.col.size() * 4, cudaMemcpyHostToDevice);
    cusparseMatDescr_t d;
    cusparseCreateMatDescr(&d);
    cusparseColorInfo_t info;
    cusparseCreateColorInfo(&info);
    const double fd = 1.0;
    const float ff = 1.0f;
    int nd = 0, nz = 0, nc = 0;
    const int sd = (int)cusparseDcsrcolor(h, g.n, (int)g.col.size(), d, (const double*)dV, dOff, dCol, &fd, &nd, dColoring, dOrder, info);
    const int sz = (int)cusparseZcsrcolor(h, g.n, (int)g.col.size(), d, (const cuDoubleComplex*)dV, dOff, dCol, &fd, &nz, dColoring, dOrder, info);
    const int sc = (int)cusparseCcsrcolor(h, g.n, (int)g.col.size(), d, (const cuComplex*)dV, dOff, dCol, &ff, &nc, dColoring, dOrder, info);
    check(sd == 0 && sz == 0 && sc == 0 && nd > 1 && nz > 1 && nc > 1, "the D, Z and C entry points color too");
    cusparseDestroyColorInfo(info);
    cusparseDestroyMatDescr(d);
    cudaFree(dOff); cudaFree(dCol); cudaFree(dV); cudaFree(dColoring); cudaFree(dOrder);
  }
  cusparseDestroy(h);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
