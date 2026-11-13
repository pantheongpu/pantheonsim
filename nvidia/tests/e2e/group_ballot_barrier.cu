// Groups of four lanes of one warp, each in a data-dependent loop of its own with a group-masked __ballot_sync, then the
// whole warp at a full-mask __shfl_down_sync and __syncthreads (#574). The shape is a cell-list neighbour build of a
// molecular dynamics code (fastMD's nl_build_kernel): each group takes one particle, walks the 27 cells around it (the
// stencil loops unrolled, so every cell has its own copy of the ballot), its lanes stride over each cell's particles,
// and the hits are compacted with the group's ballot so every lane keeps the same running count. Groups of a warp have
// different trip counts, so they wait at different copies of the ballot, and within a group the lanes take different
// paths to it (out of range, the particle itself, too far, excluded). The simulated H100 reported
//     bar.sync cannot be reached by every lane of the warp: some lanes are on a path that never arrives at this barrier
// for this kernel; the card runs it. The answer is checked against the host: every count, and every listed neighbour.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x)                                                                                    \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) {                                                                        \
      std::printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));           \
      std::exit(1);                                                                                 \
    }                                                                                               \
  } while (0)

namespace {

constexpr int kTpa = 4;      // lanes per particle
constexpr int kBlock = 256;

struct Args {
  const float4* pos;      // in cell order
  const int* order;       // cell order -> particle
  const int* cell_start;
  const int* cell_end;
  const int* excl_off;    // per particle: its exclusions are excl[excl_off[i] .. excl_off[i + 1]), or null
  const int* excl;
  int* nbr;               // nbr[slot * n + i]
  int* nnbr;
  int* max_count;
  unsigned* done;
  int* finished;
  int n, nc, max_nbr;
  float box, rc2;
};

__device__ __forceinline__ int wrap(int c, int n) {
  c %= n;
  return c < 0 ? c + n : c;
}
__device__ __forceinline__ float min_image(float d, float box) { return d - box * rintf(d / box); }

__device__ __forceinline__ bool excluded(const Args& a, int i, int j) {
  if (!a.excl_off) return false;
  for (int e = a.excl_off[i]; e < a.excl_off[i + 1]; ++e)
    if (a.excl[e] == j) return true;
  return false;
}

__global__ void build(Args a) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = g / kTpa;
  const int sub = threadIdx.x & (kTpa - 1);
  const unsigned lane = threadIdx.x & 31;
  const unsigned group = ((1u << kTpa) - 1u) << (lane & ~(kTpa - 1));
  int count = 0;
  if (k < a.n) {
    const int i = a.order[k];
    const float4 pi = a.pos[k];
    const int cx = min(int(pi.x / a.box * a.nc), a.nc - 1);
    const int cy = min(int(pi.y / a.box * a.nc), a.nc - 1);
    const int cz = min(int(pi.z / a.box * a.nc), a.nc - 1);
    const bool dedup = a.nc < 3;   // fewer than three cells a side: the periodic stencil meets a cell twice
    int visited[27];
    int n_visited = 0;
    for (int dz = -1; dz <= 1; dz++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          const int cell = (wrap(cz + dz, a.nc) * a.nc + wrap(cy + dy, a.nc)) * a.nc + wrap(cx + dx, a.nc);
          if (dedup) {
            bool seen = false;
            for (int v = 0; v < n_visited; v++) seen |= visited[v] == cell;
            if (seen) continue;
            visited[n_visited++] = cell;
          }
          const int start = a.cell_start[cell], end = a.cell_end[cell];
          for (int base = start; base < end; base += kTpa) {
            const int b = base + sub;
            int j = -1;
            if (b < end && b != k) {
              const float4 pj = a.pos[b];
              const float ex = min_image(pj.x - pi.x, a.box), ey = min_image(pj.y - pi.y, a.box),
                          ez = min_image(pj.z - pi.z, a.box);
              if (ex * ex + ey * ey + ez * ez < a.rc2) {
                j = a.order[b];
                if (excluded(a, i, j)) j = -1;
              }
            }
            const unsigned hits = __ballot_sync(group, j >= 0) & group;
            if (j >= 0) {
              const int slot = count + __popc(hits & ((1u << lane) - 1u));
              if (slot < a.max_nbr) a.nbr[slot * a.n + i] = j;
            }
            count += __popc(hits);
          }
        }
    if (sub == 0) a.nnbr[i] = count;
  }
  for (int o = 16; o > 0; o >>= 1) count = max(count, __shfl_down_sync(0xffffffffu, count, o));
  if (lane == 0) atomicMax(a.max_count, count);

  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(a.done, 1u) == gridDim.x - 1;
  __syncthreads();
  if (last && threadIdx.x == 0) {
    *a.finished = 1;
    *a.done = 0u;
  }
}

template <class T> T* to_dev(const std::vector<T>& h) {
  T* d = nullptr;
  CHECK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
  if (!h.empty()) CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
template <class T> std::vector<T> from_dev(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

double host_min_image(double d, double box) { return d - box * std::nearbyint(d / box); }

// One build: n particles in a periodic box, `clustered` of them packed into a corner (their counts run far past
// max_nbr), with or without exclusions (each particle excludes the next two). Returns the number of mismatches.
int run_case(const char* name, int n, float box, float rc, double clustered, bool with_excl, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, box), c(0.0f, 0.15f * box);
  std::vector<float4> pos(n);
  for (int i = 0; i < n; i++) {
    const bool cl = i < clustered * n;
    pos[i] = make_float4(cl ? c(rng) : u(rng), cl ? c(rng) : u(rng), cl ? c(rng) : u(rng), 0.0f);
  }
  const int nc = std::max(1, int(box / rc));
  auto cell_of = [&](const float4& p) {
    const int x = std::min(int(p.x / box * nc), nc - 1), y = std::min(int(p.y / box * nc), nc - 1),
              z = std::min(int(p.z / box * nc), nc - 1);
    return (z * nc + y) * nc + x;
  };
  const int ncells = nc * nc * nc;
  std::vector<int> order(n), start(ncells, 0), end(ncells, 0);
  for (int i = 0; i < n; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int p, int q) { return cell_of(pos[p]) < cell_of(pos[q]); });
  std::vector<float4> sorted(n);
  for (int k = 0; k < n; k++) sorted[k] = pos[order[k]];
  for (int cc = 0, k = 0; cc < ncells; cc++) {
    start[cc] = k;
    while (k < n && cell_of(sorted[k]) == cc) k++;
    end[cc] = k;
  }
  std::vector<int> excl_off, excl;
  if (with_excl) {
    excl_off.push_back(0);
    for (int i = 0; i < n; i++) {
      for (int d : {-2, -1, 1, 2})
        if (i + d >= 0 && i + d < n) excl.push_back(i + d);
      excl_off.push_back(int(excl.size()));
    }
  }
  const int max_nbr = 128;
  const float rc2 = rc * rc;

  Args a{};
  a.pos = to_dev(sorted);
  a.order = to_dev(order);
  a.cell_start = to_dev(start);
  a.cell_end = to_dev(end);
  a.excl_off = with_excl ? to_dev(excl_off) : nullptr;
  a.excl = with_excl ? to_dev(excl) : nullptr;
  a.nbr = to_dev(std::vector<int>((size_t)max_nbr * n, -1));
  a.nnbr = to_dev(std::vector<int>(n, -1));
  a.max_count = to_dev(std::vector<int>(1, 0));
  a.done = to_dev(std::vector<unsigned>(1, 0u));
  a.finished = to_dev(std::vector<int>(1, 0));
  a.n = n;
  a.nc = nc;
  a.max_nbr = max_nbr;
  a.box = box;
  a.rc2 = rc2;
  const int blocks = (n * kTpa + kBlock - 1) / kBlock;
  build<<<blocks, kBlock>>>(a);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  const auto nnbr = from_dev(a.nnbr, n);
  const auto nbr = from_dev(a.nbr, (size_t)max_nbr * n);
  const int max_count = from_dev(a.max_count, 1)[0], finished = from_dev(a.finished, 1)[0];

  // The host's lists, in double; pairs within a hair of the cut-off may fall either way in float.
  int bad = 0, over = 0, dev_max = 0;
  for (int i = 0; i < n; i++) {
    int sure = 0, edge = 0;
    for (int j = 0; j < n; j++) {
      if (j == i) continue;
      if (with_excl && std::abs(i - j) <= 2) continue;
      const double ex = host_min_image(double(pos[j].x) - pos[i].x, box), ey = host_min_image(double(pos[j].y) - pos[i].y, box),
                   ez = host_min_image(double(pos[j].z) - pos[i].z, box);
      const double d2 = ex * ex + ey * ey + ez * ez;
      if (std::fabs(d2 - rc2) < 1e-4) edge++;
      else if (d2 < rc2) sure++;
    }
    if (nnbr[i] < sure || nnbr[i] > sure + edge) {
      if (bad < 5) std::printf("  %s: particle %d has %d neighbours, the host %d (+%d at the cut-off)\n", name, i, nnbr[i], sure, edge);
      bad++;
    }
    dev_max = std::max(dev_max, nnbr[i]);
    if (nnbr[i] > max_nbr) over++;
    // The slots it filled hold distinct neighbours within the cut-off.
    std::vector<int> listed;
    for (int s = 0; s < std::min(nnbr[i], max_nbr); s++) listed.push_back(nbr[(size_t)s * n + i]);
    std::sort(listed.begin(), listed.end());
    for (size_t s = 0; s < listed.size(); s++) {
      const int j = listed[s];
      bool ok = j >= 0 && j < n && j != i && (s == 0 || listed[s - 1] != j) && !(with_excl && std::abs(i - j) <= 2);
      if (ok) {
        const double ex = host_min_image(double(pos[j].x) - pos[i].x, box), ey = host_min_image(double(pos[j].y) - pos[i].y, box),
                     ez = host_min_image(double(pos[j].z) - pos[i].z, box);
        ok = ex * ex + ey * ey + ez * ez < rc2 + 1e-4;
      }
      if (!ok) {
        if (bad < 5) std::printf("  %s: particle %d lists %d in slot order %zu, which is not a neighbour\n", name, i, j, s);
        bad++;
      }
    }
  }
  if (max_count != dev_max) {
    std::printf("  %s: max_count %d, the largest count %d\n", name, max_count, dev_max);
    bad++;
  }
  if (finished != 1) {
    std::printf("  %s: the last block did not finish\n", name);
    bad++;
  }
  std::printf("%-4s %s: %d particles, %d cells a side, largest count %d, %d past the %d slots\n", bad ? "FAIL" : "ok", name, n, nc,
              dev_max, over, max_nbr);
  for (const void* p : {(const void*)a.pos, (const void*)a.order, (const void*)a.cell_start, (const void*)a.cell_end,
                        (const void*)a.excl_off, (const void*)a.excl, (const void*)a.nbr, (const void*)a.nnbr,
                        (const void*)a.max_count, (const void*)a.done, (const void*)a.finished})
    if (p) CHECK(cudaFree(const_cast<void*>(p)));
  return bad;
}

}  // namespace

int main() {
  int bad = 0;
  // fastMD's VerletList.GrowsOnOverflowAndStaysComplete: 1500 particles, half of them in a corner, a 12 box, rc + skin 3.1.
  bad += run_case("half clustered", 1500, 12.0f, 3.1f, 0.5, false, 3);
  // The same with exclusions (a per-lane loop before the ballot), and a box of two cells a side (the de-duplicated stencil).
  bad += run_case("clustered, exclusions", 1500, 12.0f, 3.1f, 0.5, true, 5);
  bad += run_case("two cells a side", 700, 7.0f, 3.1f, 0.3, false, 7);
  bad += run_case("uniform", 2000, 15.0f, 3.1f, 0.0, true, 9);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
