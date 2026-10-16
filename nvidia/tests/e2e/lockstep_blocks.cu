// Blocks that wait on one another, in a grid that fits the device.
//
// A card keeps every block of such a grid resident at once, so a block may spin on a word another
// block writes: a hand-written grid barrier, a chain of look-back flags that runs from the last block
// to the first, a producer in the last block and consumers in the others, and RXMesh's patch locks
// (each block locks its own patch and then every other, and a block that finds a lock held by a block
// with a lower id gives up, so exactly one block ends up with all of them).
//
// The simulator used to run a grid's blocks on host threads one after another, so the first block of
// each of these spun for ever on a word a block that had not started would write, and the lock test
// gave a different answer on every run. A kernel with a loop that polls global memory now has its
// resident blocks take turns (see "Blocks that wait on one another" in the README), so each of these
// finishes with the answers a card gives. Every count here is computed on the host.
//
// Each wait is bounded (kLimit polls): a run that does not interleave the blocks fails with a count
// of timed-out waits instead of hanging.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

constexpr unsigned kLimit = 1u << 18;   // polls before a wait gives up

// ---- 1. a grid barrier built from atomics, used three times ----------------------------------------

struct Barrier {
  unsigned count;
  unsigned gen;
};

__device__ unsigned g_timeouts;

__device__ void grid_barrier(Barrier* b, unsigned nblocks) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned g = *(volatile unsigned*)&b->gen;
    __threadfence();
    if (atomicAdd(&b->count, 1u) == nblocks - 1) {
      b->count = 0;
      __threadfence();
      atomicAdd(&b->gen, 1u);
    } else {
      unsigned polls = 0;
      while (*(volatile unsigned*)&b->gen == g && ++polls < kLimit) {
      }
      if (polls >= kLimit) atomicAdd(&g_timeouts, 1u);
    }
    __threadfence();
  }
  __syncthreads();
}

// Phase p: block b writes (b + 1) * (p + 1); after the barrier every block sums what all of them
// wrote. Run one block at a time, block 0 sums zeros.
__global__ void barrier_kernel(Barrier* bar, unsigned* data, unsigned* sums, unsigned phases) {
  const unsigned n = gridDim.x;
  for (unsigned p = 0; p < phases; ++p) {
    if (threadIdx.x == 0) data[p * n + blockIdx.x] = (blockIdx.x + 1) * (p + 1);
    __threadfence();
    grid_barrier(bar, n);
    if (threadIdx.x == 0) {
      unsigned s = 0;
      for (unsigned i = 0; i < n; ++i) s += *(volatile unsigned*)&data[p * n + i];
      sums[blockIdx.x * phases + p] = s;
    }
    grid_barrier(bar, n);
  }
}

// ---- 2. a chain of flags from the last block to the first -------------------------------------------

// Block b waits for block b + 1's total, adds its own value and publishes the result. The last block
// starts the chain. A look-back that runs against the launch order: block 0 cannot finish before every
// other block has.
__global__ void reverse_chain(volatile unsigned* flag, volatile unsigned* total, unsigned* out, unsigned* timeouts) {
  if (threadIdx.x != 0) return;
  const unsigned b = blockIdx.x, n = gridDim.x;
  unsigned above = 0;
  if (b + 1 < n) {
    unsigned polls = 0;
    while (flag[b + 1] == 0 && ++polls < kLimit) {
    }
    if (polls >= kLimit) atomicAdd(timeouts, 1u);
    above = total[b + 1];
  }
  const unsigned mine = above + (b + 1) * 3;
  total[b] = mine;
  __threadfence();
  flag[b] = 1;
  out[b] = mine;
}

// ---- 3. a producer in the last block, consumers in the others ---------------------------------------

// The consumers take items from a shared ticket and wait for the producer to publish them.
__global__ void producer_consumer(volatile unsigned* ready, volatile unsigned* items, unsigned* ticket,
                                  unsigned* taken_sum, unsigned* timeouts, unsigned nitems) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == gridDim.x - 1) {
    for (unsigned i = 0; i < nitems; ++i) {
      items[i] = 100 + i;
      __threadfence();
      ready[i] = 1;
    }
    return;
  }
  for (;;) {
    const unsigned i = atomicAdd(ticket, 1u);
    if (i >= nitems) return;
    unsigned polls = 0;
    while (ready[i] == 0 && ++polls < kLimit) {
    }
    if (polls >= kLimit) {
      atomicAdd(timeouts, 1u);
      return;
    }
    atomicAdd(taken_sum, items[i]);
  }
}

// ---- 4. patch locks: exactly one block takes them all -----------------------------------------------

// The protocol of RXMesh's PatchLock. A lock word is FREE or LOCKED and a second word, `spin`, holds
// the id of the block that owns the lock or has waited longest for it. After kAttempts failed tries
// a block lowers `spin` to its own id, and gives up if it finds a lower one there.
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

// ---- 5. more blocks than the device holds -----------------------------------------------------------

// Block b waits for block b - 1 and then publishes its own value: a chain that only needs the blocks to
// start in order, as they do on a card, however many there are.
__global__ void forward_chain(volatile unsigned* flag, unsigned* out, unsigned* timeouts) {
  if (threadIdx.x != 0) return;
  const unsigned b = blockIdx.x;
  if (b > 0) {
    unsigned polls = 0;
    while (flag[b - 1] == 0 && ++polls < kLimit) {
    }
    if (polls >= kLimit) atomicAdd(timeouts, 1u);
  }
  out[b] = b * 2 + 1;
  __threadfence();
  flag[b] = 1;
}

#define CHECK(x)                                                                     \
  do {                                                                               \
    cudaError_t e_ = (x);                                                            \
    if (e_ != cudaSuccess) {                                                         \
      std::printf("FAIL: %s: %s\n", #x, cudaGetErrorString(e_));                     \
      return 1;                                                                      \
    }                                                                                \
  } while (0)

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) std::exit(1);
  cudaMemset(p, 0, n * sizeof(T));
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
  // `lockstep_blocks <name>` runs one of barrier, chain, queue, locks, forward; none runs them all.
  const auto want = [&](const char* name) { return argc < 2 || !std::strcmp(argv[1], name); };
  int dev_count = 0;
  CHECK(cudaGetDeviceCount(&dev_count));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s, %d SMs\n", prop.name, prop.multiProcessorCount);

  // 1. grid barrier: 16 blocks, three phases.
  if (want("barrier")) {
    const unsigned n = 16, phases = 3;
    Barrier* bar = dev<Barrier>(1);
    unsigned* data = dev<unsigned>(n * phases);
    unsigned* sums = dev<unsigned>(n * phases);
    unsigned zero = 0;
    CHECK(cudaMemcpyToSymbol(g_timeouts, &zero, sizeof zero));
    barrier_kernel<<<n, 64>>>(bar, data, sums, phases);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned> h(n * phases);
    CHECK(cudaMemcpy(h.data(), sums, h.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    unsigned timeouts = 0;
    CHECK(cudaMemcpyFromSymbol(&timeouts, g_timeouts, sizeof timeouts));
    bool ok = true;
    for (unsigned b = 0; b < n; ++b)
      for (unsigned p = 0; p < phases; ++p) ok = ok && h[b * phases + p] == (n * (n + 1) / 2) * (p + 1);
    std::printf("barrier: sums %s, %u timed-out waits\n", ok ? "right" : "wrong", timeouts);
    expect(ok && timeouts == 0, "grid barrier: every block sees every block's write");
    cudaFree(bar), cudaFree(data), cudaFree(sums);
  }

  // 2. a chain from the last block to the first.
  if (want("chain")) {
    const unsigned n = 24;
    unsigned* flag = dev<unsigned>(n + 1);
    unsigned* total = dev<unsigned>(n + 1);
    unsigned* out = dev<unsigned>(n);
    unsigned* timeouts = dev<unsigned>(1);
    reverse_chain<<<n, 32>>>(flag, total, out, timeouts);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned> h(n);
    unsigned t = 0;
    CHECK(cudaMemcpy(h.data(), out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&t, timeouts, sizeof t, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned b = 0; b < n; ++b) {
      unsigned want = 0;
      for (unsigned i = b; i < n; ++i) want += (i + 1) * 3;
      ok = ok && h[b] == want;
    }
    std::printf("reverse chain: %s, %u timed-out waits\n", ok ? "right" : "wrong", t);
    expect(ok && t == 0, "reverse chain: block 0 waits for all the blocks after it");
    cudaFree(flag), cudaFree(total), cudaFree(out), cudaFree(timeouts);
  }

  // 3. one producer (the last block), consumers in the others.
  if (want("queue")) {
    const unsigned n = 10, items = 40;
    unsigned* ready = dev<unsigned>(items);
    unsigned* vals = dev<unsigned>(items);
    unsigned* ticket = dev<unsigned>(1);
    unsigned* sum = dev<unsigned>(1);
    unsigned* timeouts = dev<unsigned>(1);
    producer_consumer<<<n, 32>>>(ready, vals, ticket, sum, timeouts, items);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    unsigned s = 0, t = 0;
    CHECK(cudaMemcpy(&s, sum, sizeof s, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&t, timeouts, sizeof t, cudaMemcpyDeviceToHost));
    unsigned want = 0;
    for (unsigned i = 0; i < items; ++i) want += 100 + i;
    std::printf("producer/consumer: sum %u (want %u), %u timed-out waits\n", s, want, t);
    expect(s == want && t == 0, "producer in the last block, consumers waiting for its items");
    cudaFree(ready), cudaFree(vals), cudaFree(ticket), cudaFree(sum), cudaFree(timeouts);
  }

  // 4. patch locks: 12 blocks, many shuffles.
  if (want("locks")) {
    const unsigned n = 12, rounds = 60;
    unsigned *lockw = dev<unsigned>(n), *spinw = dev<unsigned>(n);
    std::vector<unsigned> none(n, kNone);
    CHECK(cudaMemcpy(spinw, none.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
    std::vector<Lock> lh(n);
    for (unsigned i = 0; i < n; ++i) lh[i] = Lock{lockw + i, spinw + i};
    Lock* locks = dev<Lock>(n);
    CHECK(cudaMemcpy(locks, lh.data(), n * sizeof(Lock), cudaMemcpyHostToDevice));
    unsigned* bp = dev<unsigned>(n);
    unsigned* status = dev<unsigned>(n);
    std::vector<unsigned> perm(n);
    for (unsigned i = 0; i < n; ++i) perm[i] = i;
    unsigned seed = 12345;
    unsigned bad = 0, lowest_won = 0;
    for (unsigned r = 0; r < rounds; ++r) {
      for (unsigned i = n - 1; i > 0; --i) {   // a shuffle that is the same on every machine
        seed = seed * 1664525u + 1013904223u;
        std::swap(perm[i], perm[(seed >> 8) % (i + 1)]);
      }
      CHECK(cudaMemcpy(bp, perm.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
      CHECK(cudaMemset(status, 0, n * sizeof(unsigned)));
      lock_kernel<<<n, 256>>>(locks, n, bp, status);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      std::vector<unsigned> st(n);
      CHECK(cudaMemcpy(st.data(), status, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
      unsigned sum = 0, who = n;
      for (unsigned b = 0; b < n; ++b)
        if (st[b]) sum += 1, who = b;
      if (sum != 1) ++bad;
      if (who == 0) ++lowest_won;
    }
    std::printf("patch locks: %u of %u rounds did not have exactly one winner; block 0 won %u\n", bad, rounds,
                lowest_won);
    // Exactly one block wins everything every time: with every block resident, each one finds a lock
    // held, or waited for, by a lower id and gives up -- all but the lowest id still in the race.
    expect(bad == 0, "patch locks: exactly one block wins");
    cudaFree(lockw), cudaFree(spinw), cudaFree(locks), cudaFree(bp), cudaFree(status);
  }

  // 5. more blocks than the device holds, each waiting for the one before it.
  if (want("forward")) {
    int per_sm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, forward_chain, 32, 0));
    const unsigned n = per_sm * prop.multiProcessorCount * 2 + 5;
    unsigned* flag = dev<unsigned>(n);
    unsigned* out = dev<unsigned>(n);
    unsigned* timeouts = dev<unsigned>(1);
    forward_chain<<<n, 32>>>(flag, out, timeouts);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned> h(n);
    unsigned t = 0;
    CHECK(cudaMemcpy(h.data(), out, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&t, timeouts, sizeof t, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (unsigned b = 0; b < n; ++b) ok = ok && h[b] == b * 2 + 1;
    std::printf("forward chain over %u blocks (%d per SM): %s, %u timed-out waits\n", n, per_sm, ok ? "right" : "wrong", t);
    expect(ok && t == 0, "a chain across a grid larger than the device holds");
    cudaFree(flag), cudaFree(out), cudaFree(timeouts);
  }

  std::printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
