// NCCL's collectives and point-to-point operations inside captured CUDA graphs, one rank per device and per thread
// of one process. On NVIDIA's NCCL a call on a capturing stream is
// recorded, not run: it happens when the graph is launched, over whatever the graph's own kernels have written
// by then, among the same launches of the other ranks' graphs (so the ranks' graphs are launched concurrently, a thread
// each); and what the call was given (the reduction
// operator, the scalar of a pre-multiplied sum) is not needed after the capture.
//
// For each operation: it is made eagerly for two values of a counter (the references); then, per rank, a capture
// is made of a kernel that writes the rank's input from the counter in device memory followed by the call, the
// pre-multiplied operator is destroyed, the capture must not have run the call (the outputs are still zero),
// and for the two counter values the graphs of all the ranks are launched together and their outputs must match
// the references. A last eager collective after all the launches checks that the communicator's order of
// collectives is still in step (a graph launch takes its place in the order when it runs).
//
// Every value is a small integer in a float, so each result is exact. The same program runs against NVIDIA's
// libnccl on two GPUs (run_graph_capture_nccl.sh --card), which is where the expectations come from.
#include <cuda_runtime.h>
#include <nccl.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static int fails = 0;
static void expect(const std::string& what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}

static std::atomic<int> nccl_errors{0};
#define NC(x) do { const ncclResult_t r_ = (x); if (r_ != ncclSuccess) { nccl_errors++; std::printf("     %s -> %d\n", #x, (int)r_); } } while (0)

constexpr int R = 2;               // ranks, one per device and thread
constexpr size_t N = 1000;         // elements per rank per call

__global__ void fill(float* p, size_t n, const int* counter, int salt) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = static_cast<float>(static_cast<int>((i * 7 + 3 * static_cast<size_t>(*counter) + 5 * salt) % 11) - 5);
}

struct Barrier {
  std::mutex m;
  std::condition_variable cv;
  int count = 0, generation = 0;
  void wait() {
    std::unique_lock<std::mutex> l(m);
    const int g = generation;
    if (++count == R) { count = 0; ++generation; cv.notify_all(); }
    else cv.wait(l, [&] { return g != generation; });
  }
};
static Barrier barrier;

struct Rank {
  cudaStream_t st = nullptr;
  int* counter = nullptr;
  float* send = nullptr;   // 2 * N
  float* recv = nullptr;   // 2 * N
  ncclComm_t comm = nullptr;
  ncclRedOp_t premul = ncclSum;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};
static Rank rk[R];
static ncclUniqueId unique_id;

struct Case {
  const char* name;
  std::function<void(int r)> call;   // the operation for rank r, issued inside a group of its own
  bool premul = false;               // uses rk[r].premul, made before and destroyed after the capture
  int writers = 3;                   // the ranks (a bit each) whose output the call writes
};

// What each rank found, per case: captured, ran nothing, matches for the two counters.
struct Findings {
  bool captured[R] = {}, untouched[R] = {}, matches[2][R] = {};
};

static std::vector<float> read_recv(int r) {
  std::vector<float> h(2 * N);
  cudaMemcpy(h.data(), rk[r].recv, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
  return h;
}
static void write_inputs(int r) {
  fill<<<(N + 127) / 128, 128, 0, rk[r].st>>>(rk[r].send, N, rk[r].counter, r);
  fill<<<(N + 127) / 128, 128, 0, rk[r].st>>>(rk[r].send + N, N, rk[r].counter, r + 7);
}
static void issue(const Case& c, int r) {
  NC(ncclGroupStart());
  c.call(r);
  NC(ncclGroupEnd());
}

static std::vector<Findings> findings;
static bool final_ok[R] = {};
static std::vector<std::vector<float>> refs;   // [case][counter][rank] flattened

static void worker(int r, const std::vector<Case>* cases) {
  cudaSetDevice(r);
  cudaStreamCreateWithFlags(&rk[r].st, cudaStreamNonBlocking);
  cudaMalloc(&rk[r].counter, sizeof(int));
  cudaMalloc(&rk[r].send, 2 * N * sizeof(float));
  cudaMalloc(&rk[r].recv, 2 * N * sizeof(float));
  cudaMemset(rk[r].send, 0, 2 * N * sizeof(float));
  cudaMemset(rk[r].recv, 0, 2 * N * sizeof(float));
  NC(ncclCommInitRank(&rk[r].comm, R, unique_id, r));
  const int values[2] = {5, 9};
  // The copy goes through the default stream, which a non-blocking stream does not wait for: wait for it here.
  auto set_counter = [&](int v) { cudaMemcpy(rk[r].counter, &v, sizeof v, cudaMemcpyHostToDevice); cudaDeviceSynchronize(); };
  barrier.wait();
  for (size_t ci = 0; ci < cases->size(); ++ci) {
    const Case& c = (*cases)[ci];
    Findings& f = findings[ci];
    auto make_premul = [&] {
      if (!c.premul) return;
      float scale = 2.0f;   // a host scalar, copied at creation
      NC(ncclRedOpCreatePreMulSum(&rk[r].premul, &scale, ncclFloat32, ncclScalarHostImmediate, rk[r].comm));
    };
    auto drop_premul = [&] { if (c.premul) NC(ncclRedOpDestroy(rk[r].premul, rk[r].comm)); };
    // The references: eager calls.
    make_premul();
    for (int k = 0; k < 2; ++k) {
      set_counter(values[k]);
      cudaMemset(rk[r].recv, 0, 2 * N * sizeof(float));
      cudaDeviceSynchronize();
      write_inputs(r);
      issue(c, r);
      cudaStreamSynchronize(rk[r].st);
      refs[(ci * 2 + k) * R + r] = read_recv(r);
      bool nonzero = false;
      for (float v : refs[(ci * 2 + k) * R + r]) nonzero = nonzero || v != 0.0f;
      if (!nonzero && (c.writers >> r & 1)) { nccl_errors++; std::printf("     %s: the eager call wrote nothing\n", c.name); }
    }
    drop_premul();
    // The capture, one rank at a time: while a capture goes on, the other threads make no CUDA calls (a call that
    // waits on the device would invalidate it), and the recorded call does not wait for the other ranks.
    make_premul();
    cudaMemset(rk[r].recv, 0, 2 * N * sizeof(float));
    set_counter(values[0]);
    barrier.wait();
    bool ok = true;
    for (int turn = 0; turn < R; ++turn) {
      if (turn == r) {
        ok = cudaStreamBeginCapture(rk[r].st, cudaStreamCaptureModeRelaxed) == cudaSuccess;
        write_inputs(r);
        issue(c, r);
        ok = ok && cudaStreamEndCapture(rk[r].st, &rk[r].graph) == cudaSuccess && rk[r].graph &&
             cudaGraphInstantiate(&rk[r].exec, rk[r].graph, 0) == cudaSuccess;
      }
      barrier.wait();
    }
    drop_premul();   // the operator is gone before the graphs run
    f.captured[r] = ok;
    cudaDeviceSynchronize();
    {
      bool untouched = true;
      for (float v : read_recv(r)) untouched = untouched && v == 0.0f;
      f.untouched[r] = untouched;
    }
    barrier.wait();
    for (int k = 0; k < 2; ++k) {
      set_counter(values[k]);
      cudaMemset(rk[r].recv, 0, 2 * N * sizeof(float));
      cudaDeviceSynchronize();
      barrier.wait();
      if (ok) {
        cudaGraphLaunch(rk[r].exec, rk[r].st);
        cudaStreamSynchronize(rk[r].st);
        const std::vector<float> got = read_recv(r);
        f.matches[k][r] = got == refs[(ci * 2 + k) * R + r];
        if (!f.matches[k][r]) {
          size_t bad = 0, first = 0;
          const std::vector<float>& want = refs[(ci * 2 + k) * R + r];
          for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != want[i]) { if (!bad) first = i; ++bad; }
          std::printf("     %s, rank %d, counter %d: %zu of %zu elements differ, the first %zu: got %g, want %g\n", c.name, r, values[k], bad, got.size(), first, got[first], want[first]);
        }
      }
      barrier.wait();
    }
    cudaGraphExecDestroy(rk[r].exec);
    cudaGraphDestroy(rk[r].graph);
    rk[r].exec = nullptr, rk[r].graph = nullptr;
  }
  // The order of collectives is still in step after all the launches: an eager one.
  set_counter(5);
  cudaMemset(rk[r].recv, 0, 2 * N * sizeof(float));
  cudaDeviceSynchronize();
  fill<<<(N + 127) / 128, 128, 0, rk[r].st>>>(rk[r].send, N, rk[r].counter, r);
  const ncclResult_t rc = ncclAllReduce(rk[r].send, rk[r].recv, N, ncclFloat32, ncclSum, rk[r].comm, rk[r].st);
  cudaStreamSynchronize(rk[r].st);
  const std::vector<float> got = read_recv(r);
  bool ok = rc == ncclSuccess;
  if (rc != ncclSuccess) std::printf("     rank %d: ncclAllReduce -> %d (%s)\n", r, (int)rc, ncclGetLastError(rk[r].comm));
  for (size_t i = 0; i < N; ++i) {
    float want = 0;
    for (int q = 0; q < R; ++q) want += static_cast<float>(static_cast<int>((i * 7 + 3 * 5 + 5 * static_cast<size_t>(q)) % 11) - 5);
    ok = ok && got[i] == want;
  }
  final_ok[r] = ok;
}

int main() {
  int devices = 0;
  cudaGetDeviceCount(&devices);
  if (devices < R) {
    std::printf("SKIP: needs two GPUs (VGPU_DEVICE_COUNT=2 on the simulator)\n");
    return 0;
  }
  if (ncclGetUniqueId(&unique_id) != ncclSuccess) {
    std::printf("SKIP: ncclGetUniqueId failed\n");
    return 0;
  }
  int version = 0;
  ncclGetVersion(&version);
  std::printf("NCCL %d\n", version);
  auto S = [](int r) { return rk[r].send; };
  auto V = [](int r) { return rk[r].recv; };
  auto C = [](int r) { return rk[r].comm; };
  auto T = [](int r) { return rk[r].st; };
  const std::vector<Case> cases = {
      {"ncclAllReduce (sum)", [&](int r) { NC(ncclAllReduce(S(r), V(r), N, ncclFloat32, ncclSum, C(r), T(r))); }},
      {"ncclAllReduce (max)", [&](int r) { NC(ncclAllReduce(S(r), V(r), N, ncclFloat32, ncclMax, C(r), T(r))); }},
      {"ncclAllReduce (average)", [&](int r) { NC(ncclAllReduce(S(r), V(r), N, ncclFloat32, ncclAvg, C(r), T(r))); }},
      {"ncclAllReduce (in place)", [&](int r) { NC(ncclAllReduce(S(r), S(r), N, ncclFloat32, ncclSum, C(r), T(r))); cudaMemcpyAsync(V(r), S(r), N * sizeof(float), cudaMemcpyDeviceToDevice, T(r)); }},
      {"ncclBroadcast", [&](int r) { NC(ncclBroadcast(S(r), V(r), N, ncclFloat32, 1, C(r), T(r))); }},
      {"ncclReduce", [&](int r) { NC(ncclReduce(S(r), V(r), N, ncclFloat32, ncclSum, 0, C(r), T(r))); }, false, 1},
      {"ncclAllGather", [&](int r) { NC(ncclAllGather(S(r), V(r), N, ncclFloat32, C(r), T(r))); }},
      {"ncclReduceScatter", [&](int r) { NC(ncclReduceScatter(S(r), V(r), N, ncclFloat32, ncclSum, C(r), T(r))); }},
      // Receive issued before send, in a group: the group runs them together.
      {"ncclRecv and ncclSend (a group)", [&](int r) { NC(ncclRecv(V(r), N, ncclFloat32, 1 - r, C(r), T(r))); NC(ncclSend(S(r), N, ncclFloat32, 1 - r, C(r), T(r))); }},
      {"ncclAllReduce (a pre-multiplied sum, its operator destroyed after the capture)",
       [&](int r) { NC(ncclAllReduce(S(r), V(r), N, ncclFloat32, rk[r].premul, C(r), T(r))); }, true},
  };
  findings.assign(cases.size(), Findings());
  refs.assign(cases.size() * 2 * R, {});
  {
    std::vector<std::thread> threads;
    for (int r = 0; r < R; ++r) threads.emplace_back(worker, r, &cases);
    for (auto& t : threads) t.join();
  }
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const std::string name = cases[ci].name;
    const Findings& f = findings[ci];
    bool captured = true, untouched = true, m0 = true, m1 = true;
    for (int r = 0; r < R; ++r) captured = captured && f.captured[r], untouched = untouched && f.untouched[r], m0 = m0 && f.matches[0][r], m1 = m1 && f.matches[1][r];
    expect(name + ": captured", captured);
    expect(name + ": the capture ran nothing", untouched);
    expect(name + ": the launch with counter 5 matches the eager call", m0);
    expect(name + ": the launch with counter 9 matches the eager call", m1);
  }
  // The order of collectives is still in step after all the launches: an eager one, done in the ranks' threads.
  expect("an eager ncclAllReduce after the launches is right", final_ok[0] && final_ok[1]);
  expect("every NCCL call answered success", nccl_errors == 0, nccl_errors);
  for (int r = 0; r < R; ++r) {
    cudaSetDevice(r);
    cudaFree(rk[r].counter);
    cudaFree(rk[r].send);
    cudaFree(rk[r].recv);
    cudaStreamDestroy(rk[r].st);
    ncclCommDestroy(rk[r].comm);
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
