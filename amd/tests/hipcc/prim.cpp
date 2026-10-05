// AMD's hipCUB and rocThrust (over rocPRIM), unmodified, on a simulated
// MI300X and, run in wave32, a Radeon RX 7900 XTX.
// They are headers, compiled into the program, so this needs nothing of ROCm
// to run. Each result is checked against the same work done on the host,
// over 100003 elements (a prime, so no tile comes out even):
//
//   hipCUB device-wide: Reduce (sum, max), Scan (inclusive, exclusive),
//     RadixSort (pairs, and keys descending), SegmentedReduce, Select
//     (unique, and by a predicate), HistogramEven
//   hipCUB in a kernel: BlockReduce, BlockScan and WarpReduce, the warp
//     being the wave, 64 or 32 lanes
//   rocThrust: sort, sort_by_key, reduce, transform, inclusive_scan, unique,
//     count_if
//
// Built ahead of time by hipcc/build.sh, from the libraries' documented APIs.
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

namespace {

constexpr int kN = 100003;
int ok = 0, total = 0;
void check(const char* what, bool good) {
  ++total;
  ok += good;
  if (!good) std::printf("wrong: %s\n", what);
}

unsigned hash(unsigned i) { return (i * 2654435761u) ^ (i >> 7); }

template <class T>
T* device(const std::vector<T>& h) {
  T* p = nullptr;
  if (hipMalloc(&p, std::max<size_t>(h.size(), 1) * sizeof(T)) != hipSuccess ||
      hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
    return nullptr;
  return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
  std::vector<T> h(n);
  if (hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) h.clear();
  return h;
}

// Runs a hipCUB device-wide call twice, as its documentation says: once to
// ask how much temporary storage it needs, once with that storage.
template <class F>
bool cub(F f) {
  size_t bytes = 0;
  if (f(nullptr, bytes) != hipSuccess) return false;
  void* temp = nullptr;
  if (hipMalloc(&temp, std::max<size_t>(bytes, 4)) != hipSuccess) return false;
  const bool good = f(temp, bytes) == hipSuccess && hipDeviceSynchronize() == hipSuccess;
  (void)hipFree(temp);
  return good;
}

struct IsOdd {
  __host__ __device__ bool operator()(int v) const { return v & 1; }
};

constexpr int kBlock = 256;
__global__ void block_and_warp(const int* in, int* block_sum, int* scanned, int* warp_sum, int n) {
  using BlockReduce = hipcub::BlockReduce<int, kBlock>;
  using BlockScan = hipcub::BlockScan<int, kBlock>;
  using WarpReduce = hipcub::WarpReduce<int>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ typename WarpReduce::TempStorage warp_storage[kBlock / 32];
  const int i = blockIdx.x * kBlock + threadIdx.x;
  const int v = i < n ? in[i] : 0;
  const int sum = BlockReduce(reduce_storage).Sum(v);
  if (threadIdx.x == 0) block_sum[blockIdx.x] = sum;
  int prefix;
  BlockScan(scan_storage).InclusiveSum(v, prefix);
  if (i < n) scanned[i] = prefix;
  const int warp = threadIdx.x / warpSize;
  const int wsum = WarpReduce(warp_storage[warp]).Sum(v);
  if (threadIdx.x % warpSize == 0) warp_sum[blockIdx.x * (kBlock / warpSize) + warp] = wsum;
}

}  // namespace

int main() {
  std::vector<int> v(kN);
  std::vector<unsigned> keys(kN);
  std::vector<float> f(kN);
  for (int i = 0; i < kN; ++i) {
    v[i] = int(hash(i) % 1000) - 500;
    keys[i] = hash(i + 7);
    f[i] = float(int(hash(i + 3) % 20000) - 10000) / 7.0f;
  }
  int* d_v = device(v);
  int* d_out = device(std::vector<int>(kN));
  int* d_one = device(std::vector<int>(1));
  float* d_f = device(f);
  float* d_fout = device(std::vector<float>(kN));

  // Reductions.
  check("DeviceReduce::Sum", cub([&](void* t, size_t& b) { return hipcub::DeviceReduce::Sum(t, b, d_v, d_one, kN); }) &&
                                 host(d_one, 1)[0] == std::accumulate(v.begin(), v.end(), 0));
  float* d_fone = device(std::vector<float>(1));
  check("DeviceReduce::Max", cub([&](void* t, size_t& b) { return hipcub::DeviceReduce::Max(t, b, d_f, d_fone, kN); }) &&
                                 host(d_fone, 1)[0] == *std::max_element(f.begin(), f.end()));

  // Scans.
  std::vector<int> want(kN);
  std::partial_sum(v.begin(), v.end(), want.begin());
  check("DeviceScan::InclusiveSum",
        cub([&](void* t, size_t& b) { return hipcub::DeviceScan::InclusiveSum(t, b, d_v, d_out, kN); }) &&
            host(d_out, kN) == want);
  std::exclusive_scan(v.begin(), v.end(), want.begin(), 0);
  check("DeviceScan::ExclusiveSum",
        cub([&](void* t, size_t& b) { return hipcub::DeviceScan::ExclusiveSum(t, b, d_v, d_out, kN); }) &&
            host(d_out, kN) == want);

  // Radix sorts: pairs, the sort being stable; and float keys descending.
  {
    unsigned *d_k = device(keys), *d_ko = device(std::vector<unsigned>(kN));
    std::vector<int> idx(kN);
    std::iota(idx.begin(), idx.end(), 0);
    int *d_i = device(idx), *d_io = device(std::vector<int>(kN));
    std::vector<int> order = idx;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    std::vector<unsigned> sorted(kN);
    for (int i = 0; i < kN; ++i) sorted[i] = keys[order[i]];
    check("DeviceRadixSort::SortPairs",
          cub([&](void* t, size_t& b) { return hipcub::DeviceRadixSort::SortPairs(t, b, d_k, d_ko, d_i, d_io, kN); }) &&
              host(d_ko, kN) == sorted && host(d_io, kN) == order);
    std::vector<float> fs = f;
    std::sort(fs.begin(), fs.end(), std::greater<float>());
    check("DeviceRadixSort::SortKeysDescending",
          cub([&](void* t, size_t& b) { return hipcub::DeviceRadixSort::SortKeysDescending(t, b, d_f, d_fout, kN); }) &&
              host(d_fout, kN) == fs);
    (void)hipFree(d_k), (void)hipFree(d_ko), (void)hipFree(d_i), (void)hipFree(d_io);
  }

  // Segmented sums over uneven segments.
  {
    std::vector<int> offsets{0};
    while (offsets.back() < kN) offsets.push_back(std::min(kN, offsets.back() + 1 + int(hash(offsets.size()) % 3000)));
    const int segments = int(offsets.size()) - 1;
    int* d_off = device(offsets);
    int* d_seg = device(std::vector<int>(segments));
    std::vector<int> sums(segments);
    for (int s = 0; s < segments; ++s) sums[s] = std::accumulate(v.begin() + offsets[s], v.begin() + offsets[s + 1], 0);
    check("DeviceSegmentedReduce::Sum", cub([&](void* t, size_t& b) {
                                          return hipcub::DeviceSegmentedReduce::Sum(t, b, d_v, d_seg, segments, d_off,
                                                                                    d_off + 1);
                                        }) && host(d_seg, segments) == sums);
    (void)hipFree(d_off), (void)hipFree(d_seg);
  }

  // Selection: runs of equal values collapsed, and the odd values kept.
  {
    std::vector<int> runs(kN);
    for (int i = 0; i < kN; ++i) runs[i] = int(hash(i / 5) % 50);
    int* d_runs = device(runs);
    std::vector<int> uniq(runs.begin(), std::unique(runs.begin(), runs.end()));
    check("DeviceSelect::Unique",
          cub([&](void* t, size_t& b) { return hipcub::DeviceSelect::Unique(t, b, d_runs, d_out, d_one, kN); }) &&
              host(d_one, 1)[0] == int(uniq.size()) && host(d_out, uniq.size()) == uniq);
    std::vector<int> odd;
    std::copy_if(v.begin(), v.end(), std::back_inserter(odd), IsOdd());
    check("DeviceSelect::If",
          cub([&](void* t, size_t& b) { return hipcub::DeviceSelect::If(t, b, d_v, d_out, d_one, kN, IsOdd()); }) &&
              host(d_one, 1)[0] == int(odd.size()) && host(d_out, odd.size()) == odd);
    (void)hipFree(d_runs);
  }

  // A histogram of 40 even bins over [-500, 500).
  {
    int* d_hist = device(std::vector<int>(40));
    std::vector<int> hist(40, 0);
    for (int x : v) ++hist[(x + 500) / 25];
    check("DeviceHistogram::HistogramEven", cub([&](void* t, size_t& b) {
                                              return hipcub::DeviceHistogram::HistogramEven(t, b, d_v, d_hist, 41, -500,
                                                                                            500, kN);
                                            }) && host(d_hist, 40) == hist);
    (void)hipFree(d_hist);
  }

  // Block and warp collectives, in a kernel of our own.
  {
    hipDeviceProp_t p;
    const bool props = hipGetDeviceProperties(&p, 0) == hipSuccess;
    const int ws = props ? p.warpSize : 64;
    const int blocks = (kN + kBlock - 1) / kBlock, warps = blocks * (kBlock / ws);
    int *d_bs = device(std::vector<int>(blocks)), *d_ws = device(std::vector<int>(warps));
    block_and_warp<<<blocks, kBlock>>>(d_v, d_bs, d_out, d_ws, kN);
    const bool ran = props && hipDeviceSynchronize() == hipSuccess;
    std::vector<int> bs(blocks, 0), wsum(warps, 0), scanned(kN);
    for (int i = 0; i < kN; ++i) {
      bs[i / kBlock] += v[i];
      wsum[i / ws] += v[i];
      scanned[i] = (i % kBlock ? scanned[i - 1] : 0) + v[i];
    }
    check("BlockReduce", ran && host(d_bs, blocks) == bs);
    check("BlockScan", ran && host(d_out, kN) == scanned);
    check("WarpReduce, a wave wide", ran && host(d_ws, warps) == wsum);
    (void)hipFree(d_bs), (void)hipFree(d_ws);
  }

  // rocThrust.
  {
    thrust::device_vector<int> dv(v.begin(), v.end());
    std::vector<int> s = v;
    std::sort(s.begin(), s.end());
    thrust::sort(dv.begin(), dv.end());
    std::vector<int> got(kN);
    thrust::copy(dv.begin(), dv.end(), got.begin());
    check("thrust::sort", got == s);

    thrust::device_vector<unsigned> dk(keys.begin(), keys.end());
    thrust::device_vector<int> dval(kN);
    thrust::sequence(dval.begin(), dval.end());
    thrust::sort_by_key(dk.begin(), dk.end(), dval.begin());
    std::vector<int> order(kN);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    thrust::copy(dval.begin(), dval.end(), got.begin());
    check("thrust::sort_by_key", got == order);

    thrust::device_vector<int> d2(v.begin(), v.end());
    check("thrust::reduce", thrust::reduce(d2.begin(), d2.end(), 0, thrust::plus<int>()) ==
                                std::accumulate(v.begin(), v.end(), 0));
    thrust::device_vector<int> sq(kN);
    thrust::transform(d2.begin(), d2.end(), sq.begin(), thrust::negate<int>());
    thrust::copy(sq.begin(), sq.end(), got.begin());
    std::vector<int> neg(kN);
    std::transform(v.begin(), v.end(), neg.begin(), std::negate<int>());
    check("thrust::transform", got == neg);
    thrust::inclusive_scan(d2.begin(), d2.end(), sq.begin());
    thrust::copy(sq.begin(), sq.end(), got.begin());
    std::partial_sum(v.begin(), v.end(), want.begin());
    check("thrust::inclusive_scan", got == want);
    const auto end = thrust::unique(dv.begin(), dv.end());
    const std::vector<int> u(s.begin(), std::unique(s.begin(), s.end()));
    got.resize(end - dv.begin());
    thrust::copy(dv.begin(), end, got.begin());
    check("thrust::unique", got == u);
    check("thrust::count_if", thrust::count_if(d2.begin(), d2.end(), IsOdd()) == std::count_if(v.begin(), v.end(), IsOdd()));
  }

  std::printf("hipCUB and rocThrust: %d of %d match the host's\n", ok, total);
  return ok == total ? 0 : 1;
}
