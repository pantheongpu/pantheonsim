// The dropout mask of cuDNN's scaled dot-product attention (the graph API's SDPA node, built by cudnn-frontend),
// element for element. The mask is the rng_dump tensor: 1 where a probability is kept. Measured on an RTX 3060
// (cuDNN 9) by reading the dump back and bisecting the dropout probability: each element is a 16-bit number from a
// Philox4x32-7 call, kept when it is at most floor((1 - p) * 65536); a call covers eight elements and is picked by
// the element's position and the graph's seed and offset (the layout is in nvidia/src/cudnn_backend.cpp, where
// VirtualGPU reproduces it). This program holds that layout on the host, the other way round -- per element, from
// its (b, h, i, j) -- and checks the library against it on a real GPU with NVIDIA's libcudnn.so.9 (it passes) and
// on VirtualGPU, for a spread of sequence lengths (multiples of 16 or not), batches and heads, head sizes, both
// 16-bit types, probabilities, seeds, and offsets that are and are not multiples of four. It also checks that the
// output is the attention over the probabilities that mask keeps, scaled by 1 / (1 - p).
//
// Where the library's kernel does not write the dump (it leaves the tensor's cells alone for a single query row
// over 257 to 512 keys, which is all of them but the first 32 columns of every 128) those cells are not compared.
#define NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
#include <cudnn_frontend.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// ---- the mask, per element ---------------------------------------------------------------------------------------

typedef unsigned __int128 u128;

// Philox4x32 with `rounds` rounds, counter {c0..c3} as the 128-bit number n, key {k0, k1}; out[0..3].
static void philox(u128 n, uint32_t k0, uint32_t k1, int rounds, uint32_t out[4]) {
  uint32_t c[4] = {static_cast<uint32_t>(n), static_cast<uint32_t>(n >> 32), static_cast<uint32_t>(n >> 64),
                   static_cast<uint32_t>(n >> 96)};
  for (int r = 0; r < rounds; ++r) {
    const uint64_t a = 0xD2511F53ull * c[0], b = 0xCD9E8D57ull * c[2];
    const uint32_t n0 = static_cast<uint32_t>(b >> 32) ^ c[1] ^ k0, n2 = static_cast<uint32_t>(a >> 32) ^ c[3] ^ k1;
    c[0] = n0, c[1] = static_cast<uint32_t>(b), c[2] = n2, c[3] = static_cast<uint32_t>(a);
    k0 += 0x9E3779B9u, k1 += 0xBB67AE85u;
  }
  std::memcpy(out, c, 16);
}

// The 16-bit number element (i, j) of head plane bh draws.
static uint32_t element_draw(int64_t bh, int64_t i, int64_t j, int64_t Sq, uint64_t seed, int64_t offset) {
  const int64_t tile = i / 16 + ((Sq + 15) / 16) * (j / 16);
  const int64_t ri = (i % 8) / 2;
  const u128 n = (static_cast<u128>(tile) << 64) + static_cast<u128>(static_cast<__int128>((offset + ri) >> 2)) +
                 static_cast<u128>(8 * bh) + static_cast<u128>(j % 8);
  uint32_t w[4];
  philox(n, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), 7, w);
  const int half = static_cast<int>((i % 2) + 2 * ((j / 8) % 2) + 4 * ((i / 8) % 2));
  return half % 2 ? (w[half / 2] >> 16) : (w[half / 2] & 0xffffu);
}

// ---- 16-bit types --------------------------------------------------------------------------------------------------

static uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  const uint32_t m = x & 0x7fffffu;
  if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  if (e <= 0) return static_cast<uint16_t>(sign);   // the values used here are normal numbers
  uint32_t r = (static_cast<uint32_t>(e) << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
  return static_cast<uint16_t>(sign | r);
}
static float h2f(uint16_t h) {
  const int e = (h >> 10) & 0x1f;
  const uint32_t m = h & 0x3ffu, sign = (h & 0x8000u) << 16;
  uint32_t x;
  if (e == 0) {
    const float v = std::ldexp(static_cast<float>(m), -24);
    return sign ? -v : v;
  }
  x = sign | (static_cast<uint32_t>(e + 127 - 15) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
static uint16_t f2bf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}
static float bf2f(uint16_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// ---- one configuration -----------------------------------------------------------------------------------------------

struct Cfg {
  int64_t B, H, Sq, Skv, D;
  bool bf16;
  float p;
  int64_t seed, offset;
  bool partial;   // the library may leave cells of the dump unwritten
};

struct Dev {
  void* p = nullptr;
  explicit Dev(size_t n) { cudaMalloc(&p, n ? n : 1); cudaMemset(p, 0, n ? n : 1); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
};

static void run(const Cfg& c, cudnnHandle_t handle) {
  char title[160];
  std::snprintf(title, sizeof title, "b%lld h%lld sq%lld skv%lld d%lld %s p=%.2f seed=%lld offset=%lld", (long long)c.B, (long long)c.H,
                (long long)c.Sq, (long long)c.Skv, (long long)c.D, c.bf16 ? "bf16" : "half", c.p, (long long)c.seed,
                (long long)c.offset);
  const std::string name = title;
  const int64_t B = c.B, H = c.H, Sq = c.Sq, Skv = c.Skv, D = c.D;
  fe::graph::Graph g;
  const auto io = c.bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF;
  g.set_io_data_type(io).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
  auto t = [&](int64_t uid, const char* n, std::vector<int64_t> dim, std::vector<int64_t> str,
               fe::DataType_t ty = fe::DataType_t::NOT_SET) {
    auto a = fe::graph::Tensor_attributes().set_name(n).set_uid(uid).set_dim(dim).set_stride(str);
    if (ty != fe::DataType_t::NOT_SET) a.set_data_type(ty);
    return g.tensor(a);
  };
  auto q = t(1, "Q", {B, H, Sq, D}, {H * Sq * D, Sq * D, D, 1});
  auto k = t(2, "K", {B, H, Skv, D}, {H * Skv * D, Skv * D, D, 1});
  auto v = t(3, "V", {B, H, Skv, D}, {H * Skv * D, Skv * D, D, 1});
  auto opts = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(false).set_attn_scale(1.0f / 8);
  auto sd = t(5, "seed", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
  auto of = t(6, "offset", {1, 1, 1, 1}, {1, 1, 1, 1}, fe::DataType_t::INT64);
  opts.set_dropout(c.p, sd, of);
  opts.set_rng_dump(t(7, "rng_dump", {B, H, Sq, Skv}, {H * Sq * Skv, Sq * Skv, Skv, 1}, fe::DataType_t::FLOAT));
  auto [o, stats] = g.sdpa(q, k, v, opts);
  (void)stats;
  o->set_output(true).set_uid(4).set_dim({B, H, Sq, D}).set_stride({H * Sq * D, Sq * D, D, 1});
  if (!g.validate().is_good() || !g.build_operation_graph(handle).is_good() ||
      !g.create_execution_plans({fe::HeurMode_t::A}).is_good() || !g.check_support(handle).is_good() ||
      !g.build_plans(handle).is_good()) {
    if (kOnSim) expect(name + ": the graph builds", false);
    else std::printf("skip %s (no engine offered on this GPU)\n", name.c_str());
    return;
  }
  // Q and K constant, so that every probability in a row is 1 / Skv and the output is V's value times the
  // dropped probabilities' sum; V constant 0.5.
  const size_t nq = B * H * Sq * D, nk = B * H * Skv * D, nm = B * H * Sq * Skv;
  auto enc = [&](float x) { return c.bf16 ? f2bf(x) : f2h(x); };
  auto dec = [&](uint16_t x) { return c.bf16 ? bf2f(x) : h2f(x); };
  std::vector<uint16_t> hq(nq, enc(0.01f)), hk(nk, enc(0.01f)), hv(nk, enc(0.5f));
  Dev dq(nq * 2), dk(nk * 2), dv(nk * 2), dout(nq * 2), dm(nm * 4), dsd(8), dof(8);
  cudaMemcpy(dq.p, hq.data(), nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dk.p, hk.data(), nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dv.p, hv.data(), nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsd.p, &c.seed, 8, cudaMemcpyHostToDevice);
  cudaMemcpy(dof.p, &c.offset, 8, cudaMemcpyHostToDevice);
  cudaMemset(dm.p, 0xff, nm * 4);   // NaN: a cell the library does not write stays one
  int64_t wsz = 0;
  (void)g.get_workspace_size(wsz);
  Dev ws(static_cast<size_t>(wsz));
  std::unordered_map<int64_t, void*> pack{{1, dq.p}, {2, dk.p}, {3, dv.p}, {4, dout.p}, {5, dsd.p}, {6, dof.p}, {7, dm.p}};
  if (!g.execute(handle, pack, ws.p).is_good()) { expect(name + ": runs", false); return; }
  cudaDeviceSynchronize();
  std::vector<float> m(nm);
  std::vector<uint16_t> out(nq);
  cudaMemcpy(m.data(), dm.p, nm * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(out.data(), dout.p, nq * 2, cudaMemcpyDeviceToHost);

  // The mask against the layout, cell by cell.
  const float threshold = std::floor((1.0f - c.p) * 65536.0f);
  size_t written = 0, wrong = 0;
  int64_t first_bad = -1;
  std::vector<int> row_complete(static_cast<size_t>(B * H * Sq), 1);
  std::vector<int> row_kept(static_cast<size_t>(B * H * Sq), 0);
  for (int64_t bh = 0; bh < B * H; ++bh)
    for (int64_t i = 0; i < Sq; ++i)
      for (int64_t j = 0; j < Skv; ++j) {
        const size_t e = static_cast<size_t>((bh * Sq + i) * Skv + j);
        const bool is01 = m[e] == 0.0f || m[e] == 1.0f;
        const bool want = static_cast<float>(element_draw(bh, i, j, Sq, static_cast<uint64_t>(c.seed), c.offset)) <= threshold;
        if (!is01) { row_complete[static_cast<size_t>(bh * Sq + i)] = 0; continue; }
        ++written;
        row_kept[static_cast<size_t>(bh * Sq + i)] += m[e] == 1.0f;
        if ((m[e] == 1.0f) != want) { ++wrong; if (first_bad < 0) first_bad = static_cast<int64_t>(e); }
      }
  expect(name + ": the dump is written" + (c.partial ? " (in part)" : ""), c.partial ? written * 4 >= nm : written == nm,
         static_cast<double>(written) / static_cast<double>(nm));
  expect(name + ": every written cell is the layout's", wrong == 0, static_cast<double>(first_bad));

  // The output: V's value times the kept probabilities, scaled by 1 / (1 - p), in the rows whose dump is complete.
  size_t checked = 0, off = 0;
  for (int64_t bh = 0; bh < B * H && wrong == 0; ++bh)
    for (int64_t i = 0; i < Sq; ++i) {
      if (!row_complete[static_cast<size_t>(bh * Sq + i)]) continue;
      const double want = 0.5 * row_kept[static_cast<size_t>(bh * Sq + i)] / (static_cast<double>(Skv) * (1.0 - c.p));
      for (int64_t d = 0; d < D; d += (D > 8 ? D / 4 : 1)) {
        const double got = dec(out[static_cast<size_t>((bh * Sq + i) * D + d)]);
        ++checked;
        off += std::fabs(got - want) > 0.02 * want + 1e-3;
      }
    }
  if (wrong == 0 && checked) expect(name + ": the output is the attention over the kept probabilities", off == 0, static_cast<double>(off));
}

int main() {
  cudnnHandle_t handle = nullptr;
  if (!cudnn_frontend::cudnn_dlhandle) { std::printf("SKIP: no libcudnn.so.9\n"); return 0; }
  using CreateFn = cudnnStatus_t (*)(cudnnHandle_t*);
  using DestroyFn = cudnnStatus_t (*)(cudnnHandle_t);
  auto create = reinterpret_cast<CreateFn>(dlsym(cudnn_frontend::cudnn_dlhandle, "cudnnCreate"));
  auto destroy = reinterpret_cast<DestroyFn>(dlsym(cudnn_frontend::cudnn_dlhandle, "cudnnDestroy"));
  if (!create || create(&handle) != CUDNN_STATUS_SUCCESS) { std::printf("SKIP: cudnnCreate failed\n"); return 0; }
  // seeds stay below 2^53 so that the simulator, which holds tensors as doubles, reads them exactly.
  const int64_t big = (int64_t{1} << 52) + 3;
  const Cfg cfgs[] = {
      {1, 1, 16, 16, 64, false, 0.5f, 1, 0, false},         // one tile
      {1, 1, 64, 64, 64, false, 0.5f, 1, 0, false},         // 4 x 4 tiles: the tile numbering
      {1, 1, 16, 128, 64, false, 0.25f, 7, 0, false},       // wide
      {1, 1, 128, 16, 64, false, 0.75f, 7, 0, false},       // tall
      {1, 1, 48, 80, 64, false, 0.5f, 5, 0, false},         // unequal tile counts
      {1, 1, 17, 33, 64, false, 0.5f, 3, 0, false},         // partial tiles
      {1, 1, 100, 70, 32, false, 0.3f, 11, 0, false},       // partial tiles, head size 32
      {1, 1, 16, 16, 64, false, 0.5f, 1, 1, false},         // the offset not a multiple of four: row groups shift
      {1, 1, 16, 16, 64, false, 0.5f, 1, 2, false},
      {1, 1, 16, 16, 64, false, 0.5f, 1, 3, false},
      {1, 1, 16, 16, 64, false, 0.5f, 1, 4, false},
      {1, 1, 32, 32, 64, false, 0.5f, 1, 1000, false},
      {1, 1, 16, 16, 64, false, 0.5f, 1, 17179869220ll, false},   // 2^34 + 36: the counter's low word wraps
      {1, 1, 16, 16, 64, false, 0.5f, big, 12, false},      // a seed past 32 bits
      {1, 1, 16, 16, 64, false, 0.5f, 0, 0, false},         // seed 0
      {2, 3, 32, 32, 64, false, 0.5f, 9, 0, false},         // batches and heads: 8 counters apart
      {4, 8, 64, 64, 64, false, 0.4f, 3, 12, false},
      {2, 2, 128, 128, 128, true, 0.5f, 9, 0, false},       // bfloat16, head size 128
      {1, 1, 64, 64, 256, false, 0.5f, 4, 0, false},        // head size 256
      {1, 1, 1, 100, 64, false, 0.5f, 3, 0, false},         // a single query row
      {1, 1, 1, 300, 64, false, 0.5f, 3, 0, true},          // ... whose dump the library writes in part
  };
  for (const Cfg& c : cfgs) run(c, handle);
  if (destroy) destroy(handle);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
