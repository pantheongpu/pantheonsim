// cuDNN's multi-GPU batch normalization, as cudnn-frontend builds it (set_peer_stats): two GPUs, one
// thread and one cuDNN handle each, the same graph on both, the peer statistics tensors shared by the two
// executions. The forward pass computes its statistics over both GPUs' batches together; the backward pass
// does too, and hands back the gradients of the scale and bias divided by the number of GPUs (so that a
// data-parallel program's sum over GPUs gives the combined batch's gradient). Measured on two RTX 3060s with
// cuDNN 9.27, which this program also passes on: the peer tensors are pinned host memory here, since the two
// cards have no peer access (cudaDeviceCanAccessPeer is 0 on them) and the contract only needs the tensors to
// be reachable by both. The division by the number of GPUs was measured with two; with more it follows
// the same rule by assumption.
//
// On VirtualGPU the executions meet through memory or files of their own instead of through the tensors' words
// (which are cuDNN's own protocol). Threads of one process meet in memory; GPUs in separate processes are told
// apart by the peer statistics tensors being memory they share through a file, and meet through files in the
// machine directory -- checked below with two processes, one per GPU, whose tensors are a shared file mapping
// pinned with cudaHostRegister (so that both cards reach it; this is what the program does on the card too, where
// it passes), and, on the simulator and any machine where it works, CUDA IPC memory. A lone execution of a
// two-GPU graph fails after VGPU_CUDNN_PEER_TIMEOUT_MS instead of waiting forever. (NVIDIA's kernel waits for ever.)
//
// Each result is checked against a reference computed here from the same rounded inputs.
#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>
#include <cuda_runtime.h>

#include <dlfcn.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;
namespace cudnn_frontend {
void* cudnn_dlhandle = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_GLOBAL);
}

static int fails = 0;
static void expect(const std::string& what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
static const bool kOnSim = std::getenv("VGPU_GPU") != nullptr;
using TA = fe::graph::Tensor_attributes;

struct Dev {
  void* p = nullptr;
  size_t n;
  explicit Dev(size_t bytes) : n(bytes ? bytes : 1) { cudaMalloc(&p, n); cudaMemset(p, 0, n); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};
// Pinned host memory both GPUs reach: the peer statistics tensors.
struct Pinned {
  void* p = nullptr;
  explicit Pinned(size_t bytes) {
    cudaHostAlloc(&p, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    std::memset(p, 0, bytes);  // cuDNN asks for the tensors to be zeroed before each run
  }
  ~Pinned() { cudaFreeHost(p); }
  Pinned(const Pinned&) = delete;
  Pinned& operator=(const Pinned&) = delete;
};
struct Handle {
  cudnnHandle_t h = nullptr;
  Handle() { if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) h = nullptr; }
  ~Handle() { if (h) cudnnDestroy(h); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
};

// ---- half -----------------------------------------------------------------------------

static uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (m ? 0x200u : 0));
  if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return static_cast<uint16_t>(sign);
    m |= 0x800000u;
    const int shift = 14 - e;
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
    return static_cast<uint16_t>(sign | r);
  }
  uint32_t r = (static_cast<uint32_t>(e) << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
  return static_cast<uint16_t>(sign | r);
}
static float h2f(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  int32_t e = (h >> 10) & 0x1f;
  uint32_t m = h & 0x3ffu, x;
  if (e == 0) {
    if (!m) x = sign;
    else {
      e = 1;
      while (!(m & 0x400u)) m <<= 1, --e;
      m &= 0x3ffu;
      x = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (m << 13);
    }
  } else if (e == 31) {
    x = sign | 0x7f800000u | (m << 13);
  } else {
    x = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// ---- the two-GPU runs ---------------------------------------------------------------------

constexpr int kN = 4, kC = 32, kH = 8, kW = 8, kPeers = 2;
constexpr size_t kPer = static_cast<size_t>(kN) * kH * kW;  // elements per channel, per GPU

struct Shared {
  bool half = false;
  std::vector<float> x[kPeers], dy[kPeers];   // already rounded to the data type
  std::vector<float> scale, bias, run_mean, run_var, g_mean, g_inv;
  void* peers[kPeers] = {};
};
struct Out {
  std::string err;
  std::vector<float> y, mean, inv, run_mean, run_var, dx, dscale, dbias;
};

static fe::DataType_t io_type(bool half) { return half ? fe::DataType_t::HALF : fe::DataType_t::FLOAT; }
static size_t esz(bool half) { return half ? 2 : 4; }
static std::vector<uint8_t> to_bytes(bool half, const std::vector<float>& v) {
  std::vector<uint8_t> b(v.size() * esz(half));
  for (size_t i = 0; i < v.size(); ++i) {
    if (half) { const uint16_t h = f2h(v[i]); std::memcpy(&b[i * 2], &h, 2); }
    else std::memcpy(&b[i * 4], &v[i], 4);
  }
  return b;
}
static std::vector<float> from_bytes(bool half, const std::vector<uint8_t>& b) {
  std::vector<float> v(b.size() / esz(half));
  for (size_t i = 0; i < v.size(); ++i) {
    if (half) { uint16_t h; std::memcpy(&h, &b[i * 2], 2); v[i] = h2f(h); }
    else std::memcpy(&v[i], &b[i * 4], 4);
  }
  return v;
}
static void upload(void* dev, const std::vector<uint8_t>& b) { cudaMemcpy(dev, b.data(), b.size(), cudaMemcpyHostToDevice); }
static std::vector<uint8_t> download(const void* dev, size_t n) {
  std::vector<uint8_t> b(n);
  cudaMemcpy(b.data(), dev, n, cudaMemcpyDeviceToHost);
  return b;
}

static bool build(fe::graph::Graph& g, cudnnHandle_t h, std::string* err) {
  auto step = [&](const char* at, fe::error_t st) {
    if (st.is_good()) return true;
    *err = std::string(at) + ": " + st.get_message();
    return false;
  };
  return step("validate", g.validate()) && step("build_operation_graph", g.build_operation_graph(h)) &&
         step("create_execution_plans", g.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})) &&
         step("check_support", g.check_support()) && step("build_plans", g.build_plans());
}

// One GPU's forward run: batch norm in training mode with running statistics.
static void forward(int rank, const Shared& s, Out* o, int peers_in_graph = kPeers) {
  cudaSetDevice(rank);
  Handle hd;
  if (!hd.h) { o->err = "cudnnCreate"; return; }
  fe::graph::Graph g;
  g.set_io_data_type(io_type(s.half)).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
  auto X = g.tensor(TA().set_name("X").set_dim({kN, kC, kH, kW}).set_stride({kC * kH * kW, 1, kC * kW, kC}).set_uid(1));
  auto chan = [&](const char* n, int uid) {
    return g.tensor(TA().set_name(n).set_dim({1, kC, 1, 1}).set_stride({kC, 1, kC, kC}).set_data_type(fe::DataType_t::FLOAT).set_uid(uid));
  };
  auto scale = chan("scale", 2), bias = chan("bias", 3), prm = chan("prm", 4), prv = chan("prv", 5);
  std::vector<std::shared_ptr<TA>> ps;
  for (int k = 0; k < peers_in_graph; ++k)
    ps.push_back(g.tensor(TA().set_dim({peers_in_graph, 4 * kC, 1, 1}).set_stride({4 * kC, 1, 4 * kC, 4 * kC})
                              .set_data_type(fe::DataType_t::FLOAT).set_uid(10 + k)));
  auto eps = g.tensor(1e-5f);
  auto mom = g.tensor(0.3f);
  auto opt = fe::graph::Batchnorm_attributes().set_epsilon(eps).set_peer_stats(ps).set_previous_running_stats(prm, prv, mom);
  auto [Y, mean, inv, nrm, nrv] = g.batchnorm(X, scale, bias, opt);
  Y->set_output(true).set_uid(20);
  mean->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(21);
  inv->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(22);
  nrm->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(23);
  nrv->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(24);
  if (!build(g, hd.h, &o->err)) return;
  int64_t ws = 0;
  if (!g.get_workspace_size(ws).is_good()) { o->err = "get_workspace_size"; return; }
  Dev work(static_cast<size_t>(ws)), x(kPer * kC * esz(s.half)), y(kPer * kC * esz(s.half));
  Dev dsc(kC * 4), dbi(kC * 4), dprm(kC * 4), dprv(kC * 4), dm(kC * 4), di(kC * 4), dnrm(kC * 4), dnrv(kC * 4);
  upload(x.p, to_bytes(s.half, s.x[rank]));
  upload(dsc.p, to_bytes(false, s.scale));
  upload(dbi.p, to_bytes(false, s.bias));
  upload(dprm.p, to_bytes(false, s.run_mean));
  upload(dprv.p, to_bytes(false, s.run_var));
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp = {
      {1, x.p}, {2, dsc.p}, {3, dbi.p}, {4, dprm.p}, {5, dprv.p}, {20, y.p}, {21, dm.p}, {22, di.p}, {23, dnrm.p}, {24, dnrv.p}};
  for (int k = 0; k < peers_in_graph; ++k) vp[10 + k] = s.peers[k];
  auto st = g.execute(hd.h, vp, work.p);
  const cudaError_t sync = cudaDeviceSynchronize();
  if (!st.is_good()) { o->err = "execute: " + st.get_message(); return; }
  if (sync != cudaSuccess) { o->err = std::string("execute: the kernel failed: ") + cudaGetErrorString(sync); return; }
  o->y = from_bytes(s.half, download(y.p, kPer * kC * esz(s.half)));
  o->mean = from_bytes(false, download(dm.p, kC * 4));
  o->inv = from_bytes(false, download(di.p, kC * 4));
  o->run_mean = from_bytes(false, download(dnrm.p, kC * 4));
  o->run_var = from_bytes(false, download(dnrv.p, kC * 4));
}

// One GPU's backward run, from the combined batch's saved statistics.
static void backward(int rank, const Shared& s, Out* o) {
  cudaSetDevice(rank);
  Handle hd;
  if (!hd.h) { o->err = "cudnnCreate"; return; }
  fe::graph::Graph g;
  g.set_io_data_type(io_type(s.half)).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
  auto act = [&](const char* n, int uid) {
    return g.tensor(TA().set_name(n).set_dim({kN, kC, kH, kW}).set_stride({kC * kH * kW, 1, kC * kW, kC}).set_uid(uid));
  };
  auto DY = act("DY", 1), X = act("X", 2);
  auto chan = [&](const char* n, int uid) {
    return g.tensor(TA().set_name(n).set_dim({1, kC, 1, 1}).set_stride({kC, 1, kC, kC}).set_data_type(fe::DataType_t::FLOAT).set_uid(uid));
  };
  auto scale = chan("scale", 3), mean = chan("mean", 4), inv = chan("inv", 5);
  std::vector<std::shared_ptr<TA>> ps;
  for (int k = 0; k < kPeers; ++k)
    ps.push_back(g.tensor(TA().set_dim({kPeers, 4 * kC, 1, 1}).set_stride({4 * kC, 1, 4 * kC, 4 * kC})
                              .set_data_type(fe::DataType_t::FLOAT).set_uid(10 + k)));
  auto opt = fe::graph::Batchnorm_backward_attributes().set_saved_mean_and_inv_variance(mean, inv).set_peer_stats(ps);
  auto [DX, dscale, dbias] = g.batchnorm_backward(DY, X, scale, opt);
  DX->set_output(true).set_uid(20);
  dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(21);
  dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(22);
  if (!build(g, hd.h, &o->err)) return;
  int64_t ws = 0;
  if (!g.get_workspace_size(ws).is_good()) { o->err = "get_workspace_size"; return; }
  const size_t nb = kPer * kC * esz(s.half);
  Dev work(static_cast<size_t>(ws)), dx(nb), ddy(nb), ddx(nb);
  Dev dsc(kC * 4), dm(kC * 4), di(kC * 4), dds(kC * 4), ddb(kC * 4);
  upload(dx.p, to_bytes(s.half, s.x[rank]));
  upload(ddy.p, to_bytes(s.half, s.dy[rank]));
  upload(dsc.p, to_bytes(false, s.scale));
  upload(dm.p, to_bytes(false, s.g_mean));
  upload(di.p, to_bytes(false, s.g_inv));
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> vp = {
      {1, ddy.p}, {2, dx.p}, {3, dsc.p}, {4, dm.p}, {5, di.p}, {20, ddx.p}, {21, dds.p}, {22, ddb.p}};
  for (int k = 0; k < kPeers; ++k) vp[10 + k] = s.peers[k];
  auto st = g.execute(hd.h, vp, work.p);
  const cudaError_t sync = cudaDeviceSynchronize();
  if (!st.is_good()) { o->err = "execute: " + st.get_message(); return; }
  if (sync != cudaSuccess) { o->err = std::string("execute: the kernel failed: ") + cudaGetErrorString(sync); return; }
  o->dx = from_bytes(s.half, download(ddx.p, nb));
  o->dscale = from_bytes(false, download(dds.p, kC * 4));
  o->dbias = from_bytes(false, download(ddb.p, kC * 4));
}

static void make_data(Shared* s, bool half) {
  s->half = half;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto round = [&](float v) { return half ? h2f(f2h(v)) : v; };
  for (int d = 0; d < kPeers; ++d) {
    s->x[d].resize(kPer * kC);
    s->dy[d].resize(kPer * kC);
    // The GPUs' batches differ in mean, so a statistic computed on one GPU alone is plainly different.
    for (size_t i = 0; i < kPer * kC; ++i) {
      s->x[d][i] = round(nd(rng) + 3.f * static_cast<float>(d) + 0.1f * static_cast<float>(i % kC));
      s->dy[d][i] = round(nd(rng) + 0.2f * static_cast<float>(d));
    }
  }
  s->scale.resize(kC), s->bias.resize(kC), s->run_mean.resize(kC), s->run_var.resize(kC);
  for (int c = 0; c < kC; ++c) {
    s->scale[c] = 1.f + 0.02f * static_cast<float>(c);
    s->bias[c] = 0.01f * static_cast<float>(c);
    s->run_mean[c] = 0.1f * static_cast<float>(c);
    s->run_var[c] = 1.f + 0.05f * static_cast<float>(c);
  }
  // The combined batch's statistics.
  std::vector<double> m(kC, 0.0), v(kC, 0.0);
  const double total = static_cast<double>(kPeers * kPer);
  for (int d = 0; d < kPeers; ++d)
    for (size_t i = 0; i < kPer * kC; ++i) m[i % kC] += s->x[d][i];
  for (int c = 0; c < kC; ++c) m[c] /= total;
  for (int d = 0; d < kPeers; ++d)
    for (size_t i = 0; i < kPer * kC; ++i) { const double e = s->x[d][i] - m[i % kC]; v[i % kC] += e * e; }
  s->g_mean.resize(kC), s->g_inv.resize(kC);
  for (int c = 0; c < kC; ++c) {
    v[c] /= total;
    s->g_mean[c] = static_cast<float>(m[c]);
    s->g_inv[c] = static_cast<float>(1.0 / std::sqrt(v[c] + 1e-5));
  }
}

// ---- one GPU per process -----------------------------------------------------------------

enum class Via { Threads, SharedHost, Ipc };
static const char* via_name(Via v) { return v == Via::Threads ? "threads" : v == Via::SharedHost ? "processes, shared pinned file" : "processes, IPC memory"; }

static void put_vec(std::FILE* f, const std::vector<float>& v) {
  const uint64_t n = v.size();
  std::fwrite(&n, 8, 1, f);
  if (!v.empty()) std::fwrite(v.data(), 4, v.size(), f);   // data() of an empty vector may be null
}
static bool get_vec(std::FILE* f, std::vector<float>* v) {
  uint64_t n = 0;
  if (std::fread(&n, 8, 1, f) != 1 || n > (1u << 24)) return false;
  v->resize(n);
  return n == 0 || std::fread(v->data(), 4, n, f) == n;
}
static void save(const Out& o, const std::string& path) {
  std::FILE* f = std::fopen((path + ".tmp").c_str(), "wb");
  if (!f) return;
  const uint64_t n = o.err.size();
  std::fwrite(&n, 8, 1, f);
  std::fwrite(o.err.data(), 1, o.err.size(), f);
  for (const std::vector<float>* v : {&o.y, &o.mean, &o.inv, &o.run_mean, &o.run_var, &o.dx, &o.dscale, &o.dbias}) put_vec(f, *v);
  std::fclose(f);
  std::rename((path + ".tmp").c_str(), path.c_str());
}
static bool load(Out* o, const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  uint64_t n = 0;
  bool ok = std::fread(&n, 8, 1, f) == 1 && n < 4096;
  if (ok) {
    o->err.resize(n);
    ok = std::fread(&o->err[0], 1, n, f) == n;
  }
  for (std::vector<float>* v : {&o->y, &o->mean, &o->inv, &o->run_mean, &o->run_var, &o->dx, &o->dscale, &o->dbias})
    ok = ok && get_vec(f, v);
  std::fclose(f);
  return ok;
}
static bool wait_for(const std::string& path, int ms) {
  for (int i = 0; i < ms; ++i) {
    if (::access(path.c_str(), F_OK) == 0) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}
static constexpr size_t kSlot = static_cast<size_t>(kPeers) * 4 * kC * 4;   // bytes of one peer statistics tensor

// A child process: one GPU's run of one phase. The spec is "rank:backward:half:via:dir".
static int child(const char* spec) {
  int rank = 0, backward = 0, half = 0;
  char via[16] = {0}, dir[400] = {0};
  if (std::sscanf(spec, "%d:%d:%d:%15[^:]:%399s", &rank, &backward, &half, via, dir) != 5) return 2;
  const std::string d = dir;
  Shared s;
  make_data(&s, half != 0);
  Out o;
  cudaSetDevice(rank);
  void* mem = nullptr;
  void* own = nullptr;
  void* opened = nullptr;
  const int other = 1 - rank;
  if (std::string(via) == "host") {
    const int fd = ::open((d + "/peers").c_str(), O_RDWR);
    mem = fd < 0 ? MAP_FAILED : ::mmap(nullptr, kPeers * kSlot, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (fd >= 0) ::close(fd);
    if (mem == MAP_FAILED || cudaHostRegister(mem, kPeers * kSlot, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
      o.err = "mapping the shared peer statistics file";
      save(o, d + "/out" + std::to_string(rank));
      return 0;
    }
    // The device's address of each tensor: not the host's, on every machine (an RTX 3060 under WSL maps registered
    // memory elsewhere).
    for (int k = 0; k < kPeers; ++k)
      if (cudaHostGetDevicePointer(&s.peers[k], static_cast<char*>(mem) + k * kSlot, 0) != cudaSuccess) {
        o.err = "the device address of the shared peer statistics file";
        save(o, d + "/out" + std::to_string(rank));
        return 0;
      }
  } else {
    cudaMalloc(&own, kSlot);
    cudaMemset(own, 0, kSlot);
    cudaIpcMemHandle_t h;
    bool ok = cudaIpcGetMemHandle(&h, own) == cudaSuccess;
    if (ok) {
      std::FILE* f = std::fopen((d + "/h" + std::to_string(rank) + ".tmp").c_str(), "wb");
      std::fwrite(&h, sizeof h, 1, f);
      std::fclose(f);
      std::rename((d + "/h" + std::to_string(rank) + ".tmp").c_str(), (d + "/h" + std::to_string(rank)).c_str());
      ok = wait_for(d + "/h" + std::to_string(other), 30000);
    }
    if (ok) {
      cudaIpcMemHandle_t ho;
      std::FILE* f = std::fopen((d + "/h" + std::to_string(other)).c_str(), "rb");
      ok = f && std::fread(&ho, sizeof ho, 1, f) == 1;
      if (f) std::fclose(f);
      ok = ok && cudaIpcOpenMemHandle(&opened, ho, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess;
    }
    if (!ok) {
      o.err = "SKIP: CUDA IPC memory is not shared between these GPUs";
      save(o, d + "/out" + std::to_string(rank));
      cudaGetLastError();
      return 0;
    }
    s.peers[rank] = own;
    s.peers[other] = opened;
  }
  if (backward) ::backward(rank, s, &o);
  else forward(rank, s, &o);
  save(o, d + "/out" + std::to_string(rank));
  // Neither side lets go of the tensors before the other is done with them.
  std::FILE* f = std::fopen((d + "/done" + std::to_string(rank)).c_str(), "wb");
  if (f) std::fclose(f);
  wait_for(d + "/done" + std::to_string(other), 60000);
  if (opened) cudaIpcCloseMemHandle(opened);
  if (own) cudaFree(own);
  if (mem && mem != MAP_FAILED) {
    cudaHostUnregister(mem);
    ::munmap(mem, kPeers * kSlot);
  }
  return 0;
}

// Runs a phase on two processes, one per GPU; false (and `why`) if they did not both answer.
static bool run_children(Via via, bool half, bool backward, Out out[kPeers], std::string* why) {
  // Shared memory (tmpfs) where there is some: the card's driver registers (cudaHostRegister) a mapping of that, not of a file
  // on a disk or network file system.
  char shm[] = "/dev/shm/dnn-norm-XXXXXX", tmp[] = "/tmp/dnn-norm-XXXXXX";
  char* tmpl = ::access("/dev/shm", W_OK) == 0 ? shm : tmp;
  if (!mkdtemp(tmpl)) { *why = "mkdtemp"; return false; }
  const std::string d = tmpl;
  if (via == Via::SharedHost) {
    const int fd = ::open((d + "/peers").c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0 || ftruncate(fd, kPeers * kSlot) != 0) { *why = "creating the shared peer statistics file"; return false; }
    ::close(fd);   // zero: cuDNN asks for the tensors to be zeroed before each run
  }
  pid_t pids[kPeers];
  for (int r = 0; r < kPeers; ++r) {
    pids[r] = fork();
    if (pids[r] == 0) {
      char spec[512];
      std::snprintf(spec, sizeof spec, "%d:%d:%d:%s:%s", r, backward ? 1 : 0, half ? 1 : 0, via == Via::SharedHost ? "host" : "ipc", d.c_str());
      setenv("DNN_NORM_CHILD", spec, 1);
      execl("/proc/self/exe", "dnn_multigpu_norm", static_cast<char*>(nullptr));
      _exit(127);
    }
  }
  bool ok = true;
  for (int r = 0; r < kPeers; ++r) {
    int st = 0, waited = 0;
    for (;;) {
      const pid_t w = waitpid(pids[r], &st, WNOHANG);
      if (w == pids[r]) break;
      if (++waited > 120000) { kill(pids[r], SIGKILL); waitpid(pids[r], &st, 0); *why = "a process did not finish in two minutes"; ok = false; break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (ok && (!WIFEXITED(st) || WEXITSTATUS(st) != 0)) { *why = "a process exited abnormally"; ok = false; }
  }
  for (int r = 0; ok && r < kPeers; ++r)
    if (!load(&out[r], d + "/out" + std::to_string(r))) { *why = "no answer from the process of GPU " + std::to_string(r); ok = false; }
  // Clean up: the simulator's rendezvous files in the machine directory (the last process out cannot always remove
  // them, a peer still being there), the processes being gone now.
  if (kOnSim) {
    std::string root;
    if (const char* e = std::getenv("VGPU_TELEMETRY_PATH"); e && e[0]) root = e;
    else if (const char* x = std::getenv("XDG_RUNTIME_DIR"); x && x[0]) root = std::string(x) + "/vgpu-telemetry.d";
    else root = "/tmp/vgpu-telemetry-" + std::to_string(getuid()) + ".d";
    if (DIR* dd = ::opendir(root.c_str())) {
      std::vector<std::string> groups;
      while (dirent* e = ::readdir(dd))
        if (std::strncmp(e->d_name, "cudnn-peers-", 12) == 0) groups.push_back(root + "/" + e->d_name);
      ::closedir(dd);
      for (const std::string& g : groups) {
        if (DIR* gd = ::opendir(g.c_str())) {
          std::vector<std::string> files;
          while (dirent* e = ::readdir(gd))
            for (int r = 0; r < kPeers; ++r)
              if (e->d_name[0] == 'r' && std::strstr(e->d_name, ("." + std::to_string(pids[r]) + ".").c_str())) files.push_back(e->d_name);
          ::closedir(gd);
          for (const std::string& f : files) ::unlink((g + "/" + f).c_str());
        }
        ::rmdir(g.c_str());
      }
    }
  }
  for (const char* n : {"/peers", "/h0", "/h1", "/out0", "/out1", "/done0", "/done1", "/h0.tmp", "/h1.tmp", "/out0.tmp", "/out1.tmp"}) ::unlink((d + n).c_str());
  ::rmdir(d.c_str());
  return ok;
}

// Returns false when the transport cannot be had here (nothing is then reported as a failure).
static bool two_gpus(bool half, Via via) {
  const std::string tag = std::string(half ? "half" : "float") + (via == Via::Threads ? "" : std::string(", ") + via_name(via));
  Shared s;
  make_data(&s, half);
  Pinned p0(kPeers * 4 * kC * 4), p1(kPeers * 4 * kC * 4);
  s.peers[0] = p0.p, s.peers[1] = p1.p;
  const double total = static_cast<double>(kPeers * kPer);
  // ---- forward
  Out f[kPeers];
  if (via == Via::Threads) {
    std::thread t0(forward, 0, std::cref(s), &f[0], kPeers), t1(forward, 1, std::cref(s), &f[1], kPeers);
    t0.join();
    t1.join();
  } else {
    std::string why;
    if (!run_children(via, half, false, f, &why)) { expect("multi-GPU forward (" + tag + "): " + why, false); return true; }
    if (f[0].err.compare(0, 5, "SKIP:") == 0 || f[1].err.compare(0, 5, "SKIP:") == 0) {
      std::printf("skip multi-GPU (%s): %s\n", tag.c_str(), f[0].err.empty() ? f[1].err.c_str() : f[0].err.c_str());
      return false;
    }
  }
  bool ran = true;
  for (int r = 0; r < kPeers; ++r)
    if (!f[r].err.empty()) { expect("multi-GPU forward (" + tag + ") on GPU " + std::to_string(r) + ": " + f[r].err, false); ran = false; }
  if (!ran) return true;
  std::vector<double> gm(kC, 0.0), gv(kC, 0.0);
  for (int d = 0; d < kPeers; ++d)
    for (size_t i = 0; i < kPer * kC; ++i) gm[i % kC] += s.x[d][i];
  for (int c = 0; c < kC; ++c) gm[c] /= total;
  for (int d = 0; d < kPeers; ++d)
    for (size_t i = 0; i < kPer * kC; ++i) { const double e = s.x[d][i] - gm[i % kC]; gv[i % kC] += e * e; }
  for (int c = 0; c < kC; ++c) gv[c] /= total;
  if (std::getenv("DNN_NORM_DEBUG"))
    for (int r = 0; r < kPeers; ++r) std::printf("debug rank %d: mean[0..3] %g %g %g %g, want %g %g, inv[0] %g\n", r, f[r].mean[0], f[r].mean[1], f[r].mean[2], f[r].mean[3], gm[0], gm[1], f[r].inv[0]);
  for (int r = 0; r < kPeers; ++r) {
    double em = 0, ei = 0, ey = 0, erm = 0, erv = 0;
    for (int c = 0; c < kC; ++c) {
      em = std::fmax(em, std::fabs(f[r].mean[c] - gm[c]));
      ei = std::fmax(ei, std::fabs(f[r].inv[c] - 1.0 / std::sqrt(gv[c] + 1e-5)) * std::sqrt(gv[c] + 1e-5));
      erm = std::fmax(erm, std::fabs(f[r].run_mean[c] - (0.7 * s.run_mean[c] + 0.3 * gm[c])));
      erv = std::fmax(erv, std::fabs(f[r].run_var[c] - (0.7 * s.run_var[c] + 0.3 * gv[c] * total / (total - 1.0))));
    }
    for (size_t i = 0; i < kPer * kC; ++i) {
      const int c = static_cast<int>(i % kC);
      const double want = s.scale[c] * (s.x[r][i] - gm[c]) / std::sqrt(gv[c] + 1e-5) + s.bias[c];
      ey = std::fmax(ey, std::fabs(f[r].y[i] - want) / (1 + std::fabs(want)));
    }
    const std::string who = "GPU " + std::to_string(r) + " (" + tag + ")";
    expect("multi-GPU forward, " + who + ": the mean and inverse variance are the combined batch's", em < 1e-4 && ei < 1e-4, std::fmax(em, ei));
    expect("multi-GPU forward, " + who + ": the output is normalized by the combined statistics", ey < (half ? 3e-3 : 1e-4), ey);
    expect("multi-GPU forward, " + who + ": the running statistics follow the combined batch (variance unbiased)", erm < 1e-4 && erv < 1e-3, std::fmax(erm, erv));
  }
  {  // The two GPUs normalize differently if each only saw its own batch: these must not be local statistics.
    double local = 0;
    std::vector<double> lm(kC, 0.0);
    for (size_t i = 0; i < kPer * kC; ++i) lm[i % kC] += s.x[0][i];
    for (int c = 0; c < kC; ++c) local = std::fmax(local, std::fabs(f[0].mean[c] - lm[c] / static_cast<double>(kPer)));
    expect("multi-GPU forward: the statistics are not one GPU's alone", local > 0.5, local);
  }
  // ---- backward (zeroed peer tensors again, as cuDNN asks)
  if (via == Via::Threads) {
    std::memset(p0.p, 0, kPeers * 4 * kC * 4);
    std::memset(p1.p, 0, kPeers * 4 * kC * 4);
  }
  Out b[kPeers];
  if (via == Via::Threads) {
    std::thread t0(backward, 0, std::cref(s), &b[0]), t1(backward, 1, std::cref(s), &b[1]);
    t0.join();
    t1.join();
  } else {
    std::string why;
    if (!run_children(via, half, true, b, &why)) { expect("multi-GPU backward (" + tag + "): " + why, false); return true; }
  }
  ran = true;
  for (int r = 0; r < kPeers; ++r)
    if (!b[r].err.empty()) { expect("multi-GPU backward (" + tag + ") on GPU " + std::to_string(r) + ": " + b[r].err, false); ran = false; }
  if (!ran) return true;
  std::vector<double> gdb(kC, 0.0), gds(kC, 0.0);
  for (int d = 0; d < kPeers; ++d)
    for (size_t i = 0; i < kPer * kC; ++i) {
      const int c = static_cast<int>(i % kC);
      gdb[c] += s.dy[d][i];
      gds[c] += s.dy[d][i] * (s.x[d][i] - s.g_mean[c]) * s.g_inv[c];
    }
  for (int r = 0; r < kPeers; ++r) {
    double edx = 0, eds = 0, edb = 0;
    for (int c = 0; c < kC; ++c) {
      eds = std::fmax(eds, std::fabs(b[r].dscale[c] - gds[c] / kPeers) / (1 + std::fabs(gds[c])));
      edb = std::fmax(edb, std::fabs(b[r].dbias[c] - gdb[c] / kPeers) / (1 + std::fabs(gdb[c])));
    }
    for (size_t i = 0; i < kPer * kC; ++i) {
      const int c = static_cast<int>(i % kC);
      const double xh = (s.x[r][i] - s.g_mean[c]) * s.g_inv[c];
      const double want = s.scale[c] * s.g_inv[c] / total * (total * s.dy[r][i] - gdb[c] - xh * gds[c]);
      edx = std::fmax(edx, std::fabs(b[r].dx[i] - want) / (1 + std::fabs(want)));
    }
    const std::string who = "GPU " + std::to_string(r) + " (" + tag + ")";
    expect("multi-GPU backward, " + who + ": dx follows from the combined sums", edx < (half ? 3e-3 : 1e-4), edx);
    expect("multi-GPU backward, " + who + ": dscale and dbias are the combined sums over the number of GPUs", eds < 1e-3 && edb < 1e-3, std::fmax(eds, edb));
  }
  return true;
}

int main() {
  if (const char* c = std::getenv("DNN_NORM_CHILD")) return child(c);
  int devices = 0;
  cudaGetDeviceCount(&devices);
  if (devices < 2) {
    std::printf("SKIP: needs two GPUs (VGPU_DEVICE_COUNT=2 on the simulator)\n");
    return 0;
  }
  {
    Handle probe;
    if (!probe.h) {
      std::printf("SKIP: cudnnCreate failed\n");
      return 0;
    }
    std::printf("cuDNN %zu, cudnn-frontend %d.%d.%d\n", cudnnGetVersion(), CUDNN_FRONTEND_MAJOR_VERSION,
                CUDNN_FRONTEND_MINOR_VERSION, CUDNN_FRONTEND_PATCH_VERSION);
  }
  // The simulator's executions wait for each other for at most this long (a second's worth is plenty).
  if (kOnSim) setenv("VGPU_CUDNN_PEER_TIMEOUT_MS", "20000", 1);
  two_gpus(true, Via::Threads);
  two_gpus(false, Via::Threads);
  // One GPU per process: the peer statistics tensors are a shared file mapping (pinned), then CUDA IPC memory.
  two_gpus(true, Via::SharedHost);
  two_gpus(false, Via::SharedHost);
  two_gpus(false, Via::Ipc);
  if (kOnSim) {
    // A participant that never comes: NVIDIA's kernel would wait for ever, so this is not run on the card.
    setenv("VGPU_CUDNN_PEER_TIMEOUT_MS", "300", 1);
    Shared s;
    make_data(&s, false);
    Pinned p0(kPeers * 4 * kC * 4), p1(kPeers * 4 * kC * 4);
    s.peers[0] = p0.p, s.peers[1] = p1.p;
    Out o;
    forward(0, s, &o, kPeers);
    expect("a lone execution of a two-GPU graph fails instead of waiting for ever", o.err.find("execute") == 0, 0);
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
