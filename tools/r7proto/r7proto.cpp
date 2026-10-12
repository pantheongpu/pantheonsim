// Throwaway prototype harness (r7-design, NOT for merging).
// Runs one kernel through (A) the real SASS engine, (B) the closure-compiled prototype,
// alternating, and checks that device memory and every launch counter are identical.
//
//   r7proto <cubin> <scenario> [grid=G] [block=B] [iters=N] [reps=R] [fast=0|1]
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>

#include "common.hpp"
#include <dlfcn.h>
#include <sys/resource.h>
#include <sys/wait.h>

#include "engine.hpp"

using namespace vgpu;
using namespace vgpu::sass;

namespace {

struct Setup {
  std::string kernel;
  std::array<uint32_t, 3> grid{1, 1, 1}, block{1, 1, 1};
  std::vector<std::vector<uint8_t>> args;
  std::vector<std::pair<uint64_t, uint64_t>> buffers;   // (addr, bytes) to compare
};

std::map<std::string, std::string> parse(int argc, char** argv, int from) {
  std::map<std::string, std::string> kv;
  for (int i = from; i < argc; ++i) {
    std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq != std::string::npos) kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

uint64_t fill_random(MemoryManager& mem, uint64_t bytes, bool floats, uint64_t salt) {
  const uint64_t a = mem.alloc(bytes);
  std::vector<uint32_t> v(bytes / 4);
  for (uint64_t i = 0; i < v.size(); ++i) v[i] = floats ? r7::tricky_f32_bits(i + salt * 7919) : static_cast<uint32_t>((i + salt) * 2654435761u);
  mem.write(a, v.data(), bytes);
  return a;
}

Setup make_setup(const std::string& scen, MemoryManager& mem, const std::map<std::string, std::string>& kv) {
  const auto geti = [&](const char* k, long d) { auto it = kv.find(k); return it == kv.end() ? d : std::atol(it->second.c_str()); };
  Setup s;
  const uint32_t block = static_cast<uint32_t>(geti("block", 256)), grid = static_cast<uint32_t>(geti("grid", 64));
  const uint64_t threads = uint64_t{grid} * block;
  s.grid = {grid, 1, 1};
  s.block = {block, 1, 1};
  if (scen == "int_virus") {
    s.kernel = "_Z16int_virus_kerneliPji";
    const uint64_t sink = fill_random(mem, threads * 4, false, 1);
    s.args = {r7::arg<int32_t>(static_cast<int32_t>(geti("iters", 4))), r7::arg<uint64_t>(sink), r7::arg<int32_t>(static_cast<int32_t>(geti("inject", 0)))};
    s.buffers = {{sink, threads * 4}};
  } else if (scen == "vecadd") {
    s.kernel = "vecadd";
    const int n = static_cast<int>(threads);
    const uint64_t a = fill_random(mem, threads * 4, true, 1), b = fill_random(mem, threads * 4, true, 2), c = fill_random(mem, threads * 4, false, 3);
    s.args = {r7::arg<uint64_t>(a), r7::arg<uint64_t>(b), r7::arg<uint64_t>(c), r7::arg<int32_t>(n)};
    s.buffers = {{c, threads * 4}};
  } else if (scen == "ffma_loop") {
    s.kernel = "ffma_loop";
    const uint64_t out = fill_random(mem, threads * 4, false, 4);
    const float a = 0.999f, b = 0.001f;
    s.args = {r7::arg<uint64_t>(out), r7::arg<int32_t>(static_cast<int32_t>(geti("iters", 100))), r7::arg<float>(a), r7::arg<float>(b)};
    s.buffers = {{out, threads * 4}};
  } else if (scen == "int_loop") {
    s.kernel = "int_loop";
    const uint64_t out = fill_random(mem, threads * 4, false, 5);
    s.args = {r7::arg<uint64_t>(out), r7::arg<int32_t>(static_cast<int32_t>(geti("iters", 100))), r7::arg<uint32_t>(12345u)};
    s.buffers = {{out, threads * 4}};
  } else if (scen == "saxpy") {
    s.kernel = "saxpy";
    const int n = static_cast<int>(threads * geti("stride_iters", 4));
    const uint64_t y = fill_random(mem, static_cast<uint64_t>(n) * 4, true, 6), x = fill_random(mem, static_cast<uint64_t>(n) * 4, true, 7);
    s.args = {r7::arg<uint64_t>(y), r7::arg<uint64_t>(x), r7::arg<float>(2.5f), r7::arg<int32_t>(n)};
    s.buffers = {{y, static_cast<uint64_t>(n) * 4}};
  } else if (scen.rfind("ops_", 0) == 0) {
    // inputs that break arithmetic: specials (ints and floats) mixed with random bit patterns
    static const uint32_t special[] = {0u, 1u, 2u, 3u, 31u, 32u, 33u, 63u, 64u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xffffffffu, 0xfffffffeu,
                                       0x55555555u, 0xaaaaaaaau, 0xdeadbeefu, 0x0000ffffu, 0xffff0000u, 0x00000001u, 0x007fffffu, 0x00800000u,
                                       0x7f7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc00123u, 0x3f800000u, 0xbf800000u,
                                       0x3f000000u, 0x80000000u, 0x4b000000u, 0xcb000000u, 0x4f000000u, 0xcf000000u};
    const uint32_t n = grid * block;
    s.kernel = scen;
    const uint64_t kout = scen == "ops_int" ? 16 : scen == "ops_wide" ? 8 : scen == "ops_shift" ? 14 : scen == "ops_float" ? 11 : scen == "ops_cvt" ? 3 : 13;
    uint64_t in[3];
    for (int k = 0; k < 3; ++k) {
      in[k] = mem.alloc(n * 4ull);
      std::vector<uint32_t> v(n);
      for (uint32_t i = 0; i < n; ++i) {
        uint64_t x = (i + 1ull) * 0x9e3779b97f4a7c15ull + k * 0x1234567ull;
        x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 29;
        v[i] = (x & 3) < 2 ? special[(x >> 8) % (sizeof special / 4)] : static_cast<uint32_t>(x >> 16);
      }
      mem.write(in[k], v.data(), n * 4ull);
    }
    const uint64_t out = fill_random(mem, kout * n * 4ull, false, 9);
    s.args = {r7::arg<uint64_t>(in[0]), r7::arg<uint64_t>(in[1]), r7::arg<uint64_t>(in[2]), r7::arg<uint64_t>(out), r7::arg<int32_t>(static_cast<int32_t>(n))};
    s.buffers = {{out, kout * n * 4ull}};
  } else {
    throw std::runtime_error("unknown scenario " + scen);
  }
  return s;
}

bool same_stats(const exec::LaunchStats& a, const exec::LaunchStats& b, std::string* why) {
  const uint64_t* pa = reinterpret_cast<const uint64_t*>(&a);
  const uint64_t* pb = reinterpret_cast<const uint64_t*>(&b);
  for (size_t i = 0; i < exec::kCounterWords; ++i)
    if (pa[i] != pb[i]) {
      char buf[96];
      std::snprintf(buf, sizeof buf, "counter word %zu: %llu vs %llu", i, static_cast<unsigned long long>(pa[i]), static_cast<unsigned long long>(pb[i]));
      *why = buf;
      return false;
    }
  return true;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace


namespace {

struct Variant {
  std::string name;
  std::unique_ptr<r7::Env> env;
  Setup setup;
  const CubinKernel* kernel = nullptr;
  r7::Program prog;
  std::vector<double> times, walls;
  exec::LaunchStats stats;
  std::string note;
};

double child_cpu() {
  rusage ru;
  getrusage(RUSAGE_CHILDREN, &ru);
  return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: r7proto <cubin> <scenario> [grid= block= iters= reps= fast= bulk= variants=base,proto,genf,genr opt=2 isa=base|avx2]\n");
    return 2;
  }
  if (!getenv("VGPU_THREADS")) setenv("VGPU_THREADS", "1", 1);
  const auto kv = parse(argc, argv, 3);
  const auto get = [&](const char* k, const char* d) { auto it = kv.find(k); return it == kv.end() ? std::string(d) : it->second; };
  const int reps = std::atoi(get("reps", "5").c_str());
  r7::set_bulk_mem(get("bulk", "0") == "1");
  r7::set_fast_counters(get("fast", "1") != "0");
  std::vector<std::string> names;
  {
    std::stringstream ss(get("variants", "base,proto"));
    std::string t;
    while (std::getline(ss, t, ',')) names.push_back(t);
  }
  const std::string opt = get("opt", "2"), isa = get("isa", "base");
  std::vector<Variant> vs(names.size());
  exec::LaunchConfig cfg;
  r7::LaunchInfo li;
  std::vector<void*> dls;
  for (size_t i = 0; i < names.size(); ++i) {
    Variant& v = vs[i];
    v.name = names[i];
    v.env = std::make_unique<r7::Env>(argv[1]);
    v.setup = make_setup(argv[2], *v.env->mem, kv);
    for (const CubinKernel& k : v.env->mod->cubin.kernels)
      if (k.name == v.setup.kernel) v.kernel = &k;
    if (!v.kernel) {
      std::fprintf(stderr, "no kernel %s\n", v.setup.kernel.c_str());
      return 2;
    }
    cfg.grid = v.setup.grid;
    cfg.block = v.setup.block;
    li.grid = v.setup.grid;
    li.block = v.setup.block;
    li.max_steps = cfg.max_steps;
    if (v.name == "base") continue;
    li.bank0 = r7::make_bank0(*v.env->mod, *v.kernel, li, v.setup.args);
    try {
      v.prog = r7::compile(*v.env->mod, *v.kernel, li);
    } catch (const r7::Unsupported& u) {
      std::printf("UNSUPPORTED: %s\n", u.what());
      return 3;
    }
    char b[256];
    if (v.name == "proto") {
      std::snprintf(b, sizeof b, "%zu SASS -> %zu micro-ops, compile %.1f us", v.prog.n_sass, v.prog.ops.size(), v.prog.compile_seconds * 1e6);
      v.note = b;
    } else if (v.name == "genf" || v.name == "genr") {
      const bool fused = v.name == "genf";
      const double g0 = r7::wall_seconds();
      r7::Generated g = r7::generate(v.prog, fused, get("kparams", "0") == "1");
      const double g1 = r7::wall_seconds();
      const std::string base = "/tmp/claude-1002/r7d/gen_" + v.setup.kernel + (fused ? "_f" : "_r");
      {
        std::ofstream f(base + ".cpp");
        f << g.source;
      }
      const std::string cmd = "g++ -O" + opt + (isa == "avx2" ? " -mavx2 -mfma" : "") + " -ffp-contract=off -fPIC -shared -o " + base + ".so " + base + ".cpp";
      const double c0 = r7::wall_seconds(), cc0 = child_cpu();
      if (std::system(cmd.c_str()) != 0) {
        std::printf("g++ failed: %s\n", cmd.c_str());
        return 4;
      }
      const double c1 = r7::wall_seconds(), cc1 = child_cpu();
      void* h = dlopen((base + ".so").c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!h) {
        std::printf("dlopen: %s\n", dlerror());
        return 4;
      }
      dls.push_back(h);
      r7::bind(g, h);
      v.prog = std::move(g.program);
      std::snprintf(b, sizeof b, "%zu SASS -> %zu regions (%zu ops fused), %zu bytes of C++, generate %.1f ms, g++ -O%s%s %.2f s wall / %.2f s cpu",
                    v.prog.n_sass, g.regions, g.fused_ops, g.source.size(), (g1 - g0) * 1e3, opt.c_str(), isa == "avx2" ? " -mavx2 -mfma" : "", c1 - c0, cc1 - cc0);
      v.note = b;
    } else {
      std::fprintf(stderr, "unknown variant %s\n", v.name.c_str());
      return 2;
    }
  }
  for (const Variant& v : vs) if (!v.note.empty()) std::printf("%-6s %s\n", v.name.c_str(), v.note.c_str());

  for (int r = 0; r < reps; ++r)
    for (Variant& v : vs) {
      cfg.grid = v.setup.grid;
      cfg.block = v.setup.block;
      li.grid = v.setup.grid;
      li.block = v.setup.block;
      const double t0 = r7::cpu_seconds(), w0 = r7::wall_seconds();
      v.stats = v.name == "base" ? sass::launch(*v.env->mod, v.setup.kernel, cfg, v.setup.args, *v.env->mem, v.env->profile)
                                 : r7::run(v.prog, li, *v.env->mem);
      v.times.push_back(r7::cpu_seconds() - t0);
      v.walls.push_back(r7::wall_seconds() - w0);
    }

  std::string why;
  const Variant& base = vs[0];
  const double wi = static_cast<double>(base.stats.instructions);
  std::printf("warp instructions per launch: %.0f   thread instructions: %.0f\n", wi, static_cast<double>(base.stats.thread_instructions));
  const double mb = median(vs[0].times);
  bool all_ok = true;
  for (Variant& v : vs) {
    bool bits = true, cnt = true;
    for (size_t i = 0; i < v.setup.buffers.size(); ++i) {
      std::vector<uint8_t> x(base.setup.buffers[i].second), y(v.setup.buffers[i].second);
      base.env->mem->read(base.setup.buffers[i].first, x.data(), x.size());
      v.env->mem->read(v.setup.buffers[i].first, y.data(), y.size());
      bits = bits && std::memcmp(x.data(), y.data(), x.size()) == 0;
    }
    cnt = same_stats(base.stats, v.stats, &why);
    const double m = median(v.times);
    std::printf("%-6s wall s/launch median %.4f  ", v.name.c_str(), median(v.walls));
    std::printf("%-6s cpu s/launch median %.4f (min %.4f max %.4f)  %7.1f M warp-inst/s  vs base %5.2fx  same bits %s  same counters %s %s\n",
                v.name.c_str(), m, *std::min_element(v.times.begin(), v.times.end()), *std::max_element(v.times.begin(), v.times.end()),
                wi / m / 1e6, mb / m, bits ? "YES" : "NO", cnt ? "YES" : "NO", cnt ? "" : why.c_str());
    all_ok = all_ok && bits && cnt;
  }
  return all_ok ? 0 : 1;
}
