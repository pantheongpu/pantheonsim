// Work-groups that wait on one another, in a grid that fits the device.
//
// A card keeps every work-group of such a grid resident at once, so a group may spin on a word another
// group writes: a hand-written grid barrier, a chain of flags that runs from the last group to the first,
// a producer in the last group and consumers in the others, and locks taken by groups that all start
// together (the protocol of RXMesh's patch locks: a group that finds a lock held, or waited for, by a group
// with a lower id gives up, so exactly one group ends up with all of them).
//
// The simulator used to run a grid's groups on host threads one after another, so the first group of each
// of these spun for ever on a group that had not started, or the lock test gave a different answer on every
// run. A kernel with a loop that polls global memory now has the groups the device holds take turns (see
// "Blocks that wait on one another" in the README). Every count is computed on the host. Each wait is
// bounded (kLimit polls), so a run that does not interleave the groups fails with a count of timed-out
// waits instead of hanging.
//
// Built by build.sh with hipcc; run by amd/tests/e2e/run_hipcc.sh. `lockstep <name>` runs one of barrier,
// chain, queue, locks, forward.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

constexpr unsigned kLimit = 1u << 18;   // polls before a wait gives up

__device__ unsigned load_relaxed(const unsigned* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

// ---- 1. a grid barrier built from atomics, used three times --------------------------------------

struct Barrier {
  unsigned count;
  unsigned gen;
};

__device__ unsigned g_timeouts;

__device__ void grid_barrier(Barrier* b, unsigned groups) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned g = load_relaxed(&b->gen);
    __threadfence();
    if (atomicAdd(&b->count, 1u) == groups - 1) {
      b->count = 0;
      __threadfence();
      atomicAdd(&b->gen, 1u);
    } else {
      unsigned polls = 0;
      while (load_relaxed(&b->gen) == g && ++polls < kLimit) {
      }
      if (polls >= kLimit) atomicAdd(&g_timeouts, 1u);
    }
    __threadfence();
  }
  __syncthreads();
}

// Phase p: group b writes (b + 1) * (p + 1); after the barrier every group sums what all of them wrote.
// Run one group at a time, group 0 sums zeros.
__global__ void barrier_kernel(Barrier* bar, unsigned* data, unsigned* sums, unsigned phases) {
  const unsigned n = gridDim.x;
  for (unsigned p = 0; p < phases; ++p) {
    if (threadIdx.x == 0) data[p * n + blockIdx.x] = (blockIdx.x + 1) * (p + 1);
    __threadfence();
    grid_barrier(bar, n);
    if (threadIdx.x == 0) {
      unsigned s = 0;
      for (unsigned i = 0; i < n; ++i) s += load_relaxed(&data[p * n + i]);
      sums[blockIdx.x * phases + p] = s;
    }
    grid_barrier(bar, n);
  }
}

// ---- 2. a chain of flags from the last group to the first ----------------------------------------

__global__ void reverse_chain(unsigned* flag, unsigned* total, unsigned* out, unsigned* timeouts) {
  if (threadIdx.x != 0) return;
  const unsigned b = blockIdx.x, n = gridDim.x;
  unsigned above = 0;
  if (b + 1 < n) {
    unsigned polls = 0;
    while (load_relaxed(&flag[b + 1]) == 0 && ++polls < kLimit) {
    }
    if (polls >= kLimit) atomicAdd(timeouts, 1u);
    above = load_relaxed(&total[b + 1]);
  }
  const unsigned mine = above + (b + 1) * 3;
  __atomic_store_n(&total[b], mine, __ATOMIC_RELAXED);
  __threadfence();
  __atomic_store_n(&flag[b], 1u, __ATOMIC_RELAXED);
  out[b] = mine;
}

// ---- 3. a producer in the last group, consumers in the others ------------------------------------

__global__ void producer_consumer(unsigned* ready, unsigned* items, unsigned* ticket, unsigned* taken_sum,
                                  unsigned* timeouts, unsigned nitems) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == gridDim.x - 1) {
    for (unsigned i = 0; i < nitems; ++i) {
      __atomic_store_n(&items[i], 100 + i, __ATOMIC_RELAXED);
      __threadfence();
      __atomic_store_n(&ready[i], 1u, __ATOMIC_RELAXED);
    }
    return;
  }
  for (;;) {
    const unsigned i = atomicAdd(ticket, 1u);
    if (i >= nitems) return;
    unsigned polls = 0;
    while (load_relaxed(&ready[i]) == 0 && ++polls < kLimit) {
    }
    if (polls >= kLimit) {
      atomicAdd(timeouts, 1u);
      return;
    }
    atomicAdd(taken_sum, load_relaxed(&items[i]));
  }
}

// ---- 4. locks: exactly one group takes them all --------------------------------------------------

struct Lock {
  unsigned* lock;
  unsigned* spin;
};

constexpr unsigned kFree = 0, kLocked = 0xffffffffu, kAttempts = 10, kNone = 0xffffffffu;

__device__ bool acquire(Lock l, unsigned id) {
  int attempt = 0;
  while (atomicCAS(l.lock, kFree, kLocked) == kLocked) {
    __threadfence();
    if (attempt == (int)kAttempts) {
      const unsigned other = atomicMin(l.spin, id);
      __threadfence();
      if (other < id) return false;
      attempt = 0;
    }
    attempt++;
  }
  atomicExch(l.spin, id);
  return true;
}

__device__ void release(Lock l) {
  atomicExch(l.spin, kNone);
  atomicExch(l.lock, kFree);
  __threadfence();
}

__global__ void lock_kernel(Lock* locks, unsigned npatches, const unsigned* block_patch, unsigned* status) {
  if (threadIdx.x != 0) return;
  const unsigned b = blockIdx.x, mine = block_patch[b];
  bool ok = acquire(locks[mine], b);
  if (ok) {
    for (unsigned p = 0; p < npatches; ++p) {
      if (p == mine) continue;
      ok = acquire(locks[p], b);
      if (!ok) {
        release(locks[mine]);
        for (unsigned q = 0; q < p; ++q)
          if (q != mine) release(locks[q]);
        break;
      }
    }
  }
  if (ok) {
    release(locks[mine]);
    for (unsigned p = 0; p < npatches; ++p)
      if (p != mine) release(locks[p]);
  }
  status[b] = ok;
}

// ---- 5. more groups than the device holds, each waiting for the one before it --------------------

__global__ void forward_chain(unsigned* flag, unsigned* out, unsigned* timeouts) {
  if (threadIdx.x != 0) return;
  const unsigned b = blockIdx.x;
  if (b > 0) {
    unsigned polls = 0;
    while (load_relaxed(&flag[b - 1]) == 0 && ++polls < kLimit) {
    }
    if (polls >= kLimit) atomicAdd(timeouts, 1u);
  }
  out[b] = b * 2 + 1;
  __threadfence();
  __atomic_store_n(&flag[b], 1u, __ATOMIC_RELAXED);
}

#define CHECK(x)                                                                \
  do {                                                                          \
    hipError_t e_ = (x);                                                        \
    if (e_ != hipSuccess) {                                                     \
      std::printf("FAIL: %s: %s\n", #x, hipGetErrorString(e_));                 \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  if (hipMalloc(&p, n * sizeof(T)) != hipSuccess) std::exit(1);
  (void)hipMemset(p, 0, n * sizeof(T));
  return p;
}

static int fails = 0;
static void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++fails;
  }
}

int main(int argc, char** argv) {
  const auto want = [&](const char* name) { return argc < 2 || !std::strcmp(argv[1], name); };
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("device %s, %d compute units\n", prop.name, prop.multiProcessorCount);

  if (want("barrier")) {
    const unsigned n = 16, phases = 3;
    Barrier* bar = dev<Barrier>(1);
    unsigned* data = dev<unsigned>(n * phases);
    unsigned* sums = dev<unsigned>(n * phases);
    unsigned zero = 0;
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_timeouts), &zero, sizeof zero));
    barrier_kernel<<<n, 64>>>(bar, data, sums, phases);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned> h(n * phases);
    CHECK(hipMemcpy(h.data(), sums, h.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
    unsigned timeouts = 0;
    CHECK(hipMemcpyFromSymbol(&timeouts, HIP_SYMBOL(g_timeouts), sizeof timeouts));
    bool ok = true;
    for (unsigned b = 0; b < n; ++b)
      for (unsigned p = 0; p < phases; ++p) ok = ok && h[b * phases + p] == (n * (n + 1) / 2) * (p + 1);
    std::printf("barrier: sums %s, %u timed-out waits\n", ok ? "right" : "wrong", timeouts);
    expect(ok && timeouts == 0, "grid barrier: every group sees every group's write");
  }

  if (want("chain")) {
    const unsigned n = 24;
    unsigned* flag = dev<unsigned>(n + 1);
    unsigned* total = dev<unsigned>(n + 1);
    unsigned* out = dev<unsigned>(n);
    unsigned* timeouts = dev<unsigned>(1);
    reverse_chain<<<n, 64>>>(flag, total, out, timeouts);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned> h(n);
    unsigned t = 0;
    CHECK(hipMemcpy(h.data(), out, n * sizeof(unsigned), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&t, timeouts, sizeof t, hipMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned b = 0; b < n; ++b) {
      unsigned want_sum = 0;
      for (unsigned i = b; i < n; ++i) want_sum += (i + 1) * 3;
      ok = ok && h[b] == want_sum;
    }
    std::printf("reverse chain: %s, %u timed-out waits\n", ok ? "right" : "wrong", t);
    expect(ok && t == 0, "reverse chain: group 0 waits for all the groups after it");
  }

  if (want("queue")) {
    const unsigned n = 10, items = 40;
    unsigned* ready = dev<unsigned>(items);
    unsigned* vals = dev<unsigned>(items);
    unsigned* ticket = dev<unsigned>(1);
    unsigned* sum = dev<unsigned>(1);
    unsigned* timeouts = dev<unsigned>(1);
    producer_consumer<<<n, 64>>>(ready, vals, ticket, sum, timeouts, items);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    unsigned s = 0, t = 0;
    CHECK(hipMemcpy(&s, sum, sizeof s, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&t, timeouts, sizeof t, hipMemcpyDeviceToHost));
    unsigned want_sum = 0;
    for (unsigned i = 0; i < items; ++i) want_sum += 100 + i;
    std::printf("producer/consumer: sum %u (want %u), %u timed-out waits\n", s, want_sum, t);
    expect(s == want_sum && t == 0, "producer in the last group, consumers waiting for its items");
  }

  if (want("locks")) {
    const unsigned n = 12, rounds = 60;
    unsigned *lockw = dev<unsigned>(n), *spinw = dev<unsigned>(n);
    std::vector<unsigned> none(n, kNone);
    CHECK(hipMemcpy(spinw, none.data(), n * sizeof(unsigned), hipMemcpyHostToDevice));
    std::vector<Lock> lh(n);
    for (unsigned i = 0; i < n; ++i) lh[i] = Lock{lockw + i, spinw + i};
    Lock* locks = dev<Lock>(n);
    CHECK(hipMemcpy(locks, lh.data(), n * sizeof(Lock), hipMemcpyHostToDevice));
    unsigned* bp = dev<unsigned>(n);
    unsigned* status = dev<unsigned>(n);
    std::vector<unsigned> perm(n);
    for (unsigned i = 0; i < n; ++i) perm[i] = i;
    unsigned seed = 12345, bad = 0;
    for (unsigned r = 0; r < rounds; ++r) {
      for (unsigned i = n - 1; i > 0; --i) {   // a shuffle that is the same on every machine
        seed = seed * 1664525u + 1013904223u;
        std::swap(perm[i], perm[(seed >> 8) % (i + 1)]);
      }
      CHECK(hipMemcpy(bp, perm.data(), n * sizeof(unsigned), hipMemcpyHostToDevice));
      CHECK(hipMemset(status, 0, n * sizeof(unsigned)));
      lock_kernel<<<n, 256>>>(locks, n, bp, status);
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());
      std::vector<unsigned> st(n);
      CHECK(hipMemcpy(st.data(), status, n * sizeof(unsigned), hipMemcpyDeviceToHost));
      unsigned sum = 0;
      for (unsigned b = 0; b < n; ++b) sum += st[b] ? 1 : 0;
      if (sum != 1) ++bad;
    }
    std::printf("locks: %u of %u rounds did not have exactly one winner\n", bad, rounds);
    expect(bad == 0, "locks: exactly one group wins");
  }

  if (want("forward")) {
    // 32 KB of LDS a group, so a compute unit holds two and the device 2 x its compute units.
    const size_t lds = 32 * 1024;
    int per_cu = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, forward_chain, 64, lds));
    const unsigned n = per_cu * prop.multiProcessorCount * 2 + 5;
    unsigned* flag = dev<unsigned>(n);
    unsigned* out = dev<unsigned>(n);
    unsigned* timeouts = dev<unsigned>(1);
    forward_chain<<<n, 64, lds>>>(flag, out, timeouts);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned> h(n);
    unsigned t = 0;
    CHECK(hipMemcpy(h.data(), out, n * sizeof(unsigned), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&t, timeouts, sizeof t, hipMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned b = 0; b < n; ++b) ok = ok && h[b] == b * 2 + 1;
    std::printf("forward chain over %u groups (%d a compute unit): %s, %u timed-out waits\n", n, per_cu, ok ? "right" : "wrong", t);
    expect(ok && t == 0, "a chain across a grid larger than the device holds");
  }

  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
