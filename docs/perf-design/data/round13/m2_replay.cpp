// M2 (host-gpu spike): replay launches captured from the simulator (VGPU_HOSTGPU_CAPTURE) on the real card and compare the
// card's memory with the simulator's, byte for byte. The memory is placed at the simulator's own virtual addresses with the
// VMM API (2 MiB granules); the module is the program's own fatbin, taken from its .nv_fatbin section.
// Build: g++ -O2 -std=c++17 m2_replay.cpp -o m2_replay -I/usr/local/cuda-13.0/include -lcuda
// Usage: m2_replay <program> <capture dir> [-v]
#include <cuda.h>
#include <elf.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <time.h>
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s_ = "?"; cuGetErrorName(r_, &s_); fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s_); return false; } } while (0)
static const uint64_t kGran = 2ull << 20, kChunk = 64ull << 10;
struct Alloc { uint64_t base, size; std::vector<uint8_t> bytes; };
static std::vector<uint8_t> slurp(const std::string& p, bool* ok) { std::vector<uint8_t> v; FILE* f = fopen(p.c_str(), "rb"); *ok = f; if (!f) return v; fseek(f, 0, SEEK_END); v.resize(ftell(f)); fseek(f, 0, SEEK_SET); if (fread(v.data(), 1, v.size(), f) != v.size()) *ok = false; fclose(f); return v; }
static bool parse_mem(const std::vector<uint8_t>& raw, std::vector<Alloc>* out) {
  size_t at = 0;
  while (at + 16 <= raw.size()) { Alloc a; memcpy(&a.base, &raw[at], 8); memcpy(&a.size, &raw[at + 8], 8); at += 16; if (at + a.size > raw.size()) return false; a.bytes.assign(raw.begin() + at, raw.begin() + at + a.size); at += a.size; out->push_back(std::move(a)); }
  return true;
}
// the fatbins of an ELF program: every 0xBA55ED50 header in .nv_fatbin
static std::vector<std::vector<uint8_t>> fatbins(const std::string& prog) {
  bool ok; std::vector<uint8_t> f = slurp(prog, &ok); std::vector<std::vector<uint8_t>> out; if (!ok) return out;
  auto* eh = (Elf64_Ehdr*)f.data(); auto* sh = (Elf64_Shdr*)(f.data() + eh->e_shoff); const char* names = (const char*)f.data() + sh[eh->e_shstrndx].sh_offset;
  for (int i = 0; i < eh->e_shnum; ++i) if (!strcmp(names + sh[i].sh_name, ".nv_fatbin")) {
    const uint8_t* b = f.data() + sh[i].sh_offset; size_t n = sh[i].sh_size;
    for (size_t o = 0; o + 16 <= n;) { uint32_t magic; memcpy(&magic, b + o, 4); if (magic != 0xBA55ED50u) { o += 8; continue; }
      uint16_t hdr; uint64_t sz; memcpy(&hdr, b + o + 6, 2); memcpy(&sz, b + o + 8, 8); size_t total = hdr + sz; if (o + total > n) total = n - o; out.emplace_back(b + o, b + o + total); o += (total + 7) & ~size_t(7); }
  }
  return out;
}
struct Run { std::vector<Alloc> res; double loop_sync_s = 0, loop_queued_s = 0; double kernel_s = 0, push_s = 0, pull_s = 0; bool ok = false; };
static std::vector<std::vector<uint8_t>> g_fbs;
static CUmodule g_mod;
struct Run;
static Run* g_loop_run = nullptr;
static bool replay(const std::string& kname, const unsigned grid[3], const unsigned block[3], unsigned shared, const std::vector<std::vector<uint8_t>>& args, const std::vector<Alloc>& pre, Run* run, CUcontext ctx) {
  // granule-aligned union of the allocations, grouped into runs of touching granules
  std::map<uint64_t, uint64_t> gr;  // start -> end (granule aligned)
  for (const Alloc& a : pre) { uint64_t s = a.base & ~(kGran - 1), e = (a.base + a.size + kGran - 1) & ~(kGran - 1); auto it = gr.upper_bound(s); if (it != gr.begin()) { auto p = std::prev(it); if (p->second >= s) { s = p->first; e = std::max(e, p->second); it = gr.erase(p); } } while (it != gr.end() && it->first <= e) { e = std::max(e, it->second); it = gr.erase(it); } gr[s] = e; }
  CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
  struct Map { CUdeviceptr va; uint64_t size; CUmemGenericAllocationHandle h; };
  std::vector<Map> maps; bool good = true;
  // One reservation over the whole span, rounded to 32 MiB: the driver honours a fixed hint only for reservations of 32 MiB or more
  // (smaller ones are placed by the driver, see 09-host-gpu-spike.md), and physical memory is mapped only where there are allocations.
  const uint64_t kBlock = 32ull << 20;
  uint64_t rs = gr.begin()->first & ~(kBlock - 1), re = (gr.rbegin()->second + kBlock - 1) & ~(kBlock - 1);
  CUdeviceptr rp = 0; { CUresult r = cuMemAddressReserve(&rp, re - rs, 0, rs, 0); if (r || rp != rs) { fprintf(stderr, "reserve at 0x%llx size %llu MiB failed (%d, got 0x%llx)\n", (unsigned long long)rs, (unsigned long long)((re - rs) >> 20), (int)r, (unsigned long long)rp); good = false; } }
  for (auto& [s, e] : gr) {
    if (!good) break;
    Map m{s, e - s, 0}; if (cuMemCreate(&m.h, e - s, &prop, 0)) { good = false; break; }
    if (cuMemMap(s, e - s, 0, m.h, 0)) { good = false; cuMemRelease(m.h); break; }
    CUmemAccessDesc ad = {}; ad.location = prop.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; cuMemSetAccess(s, e - s, &ad, 1);
    cuMemsetD8(s, 0, e - s); maps.push_back(m);
  }
  if (good) {
    double t0 = now();
    for (const Alloc& a : pre) if (cuMemcpyHtoD(a.base, a.bytes.data(), a.size)) { good = false; break; }
    run->push_s = now() - t0;
  }
  if (good) {
    CUfunction fn = nullptr; CUmodule mod = nullptr;
    for (auto& fb : g_fbs) { if (cuModuleLoadData(&mod, fb.data()) == CUDA_SUCCESS) { if (cuModuleGetFunction(&fn, mod, kname.c_str()) == CUDA_SUCCESS) break; cuModuleUnload(mod); mod = nullptr; fn = nullptr; } }
    if (!fn) { fprintf(stderr, "no function %s in the program's fatbins\n", kname.c_str()); good = false; }
    else {
      if (shared > 48 * 1024) cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared);
      std::vector<void*> kp; std::vector<std::vector<uint8_t>> keep = args; for (auto& a : keep) kp.push_back(a.data());
      cuCtxSynchronize(); double t1 = now();
      CUresult r = cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, 0, kp.data(), nullptr);
      CUresult s = r ? r : cuCtxSynchronize(); run->kernel_s = now() - t1;
      if (!s && getenv("M2_LOOP") && run == g_loop_run) {   // data already resident: launch + sync, back to back, and launches queued then one sync
        int n = atoi(getenv("M2_LOOP")); double ta = now();
        for (int i = 0; i < n; ++i) { cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, 0, kp.data(), nullptr); cuCtxSynchronize(); }
        run->loop_sync_s = (now() - ta) / n; ta = now();
        for (int i = 0; i < n; ++i) cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, 0, kp.data(), nullptr);
        cuCtxSynchronize(); run->loop_queued_s = (now() - ta) / n;
      }
      if (s) { const char* n = "?"; cuGetErrorName(s, &n); fprintf(stderr, "launch failed: %s\n", n); good = false; }
    }
    if (mod) {
      // kept until the results are read: the module's own memory must not move
    }
    g_mod = mod;
  }
  if (good) {
    double t2 = now();
    for (const Alloc& a : pre) { Alloc o; o.base = a.base; o.size = a.size; o.bytes.resize(a.size); if (cuMemcpyDtoH(o.bytes.data(), a.base, a.size)) { good = false; break; } run->res.push_back(std::move(o)); }
    run->pull_s = now() - t2;
  }
  if (g_mod) { cuModuleUnload(g_mod); g_mod = nullptr; }
  for (auto& m : maps) { cuMemUnmap(m.va, m.size); cuMemRelease(m.h); }
  if (rp) cuMemAddressFree(rp, re - rs);
  run->ok = good; return good;
}
static float ulps(uint32_t a, uint32_t b) { auto key = [](uint32_t x) { return (int64_t)((x & 0x80000000u) ? -(int64_t)(x & 0x7fffffffu) : (int64_t)x); }; return (float)llabs(key(a) - key(b)); }
int main(int argc, char** argv) {
  if (argc < 3) return 2; std::string prog = argv[1], dir = argv[2]; bool verbose = argc > 3;
  cuInit(0); CUdevice d; cuDeviceGet(&d, 0); CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, d); cuCtxSetCurrent(ctx);
  auto& fbs = g_fbs; fbs = fatbins(prog); if (fbs.empty()) { printf("no .nv_fatbin in %s\n", prog.c_str()); return 3; }
  int agree = 0, differ = 0, nondet = 0, fail = 0;
  for (int seq = 0;; ++seq) {
    char base[64]; snprintf(base, sizeof base, "%s/%06d", dir.c_str(), seq); bool ok;
    std::string meta_s; { auto m = slurp(std::string(base) + ".meta", &ok); if (!ok) break; meta_s.assign(m.begin(), m.end()); }
    char kname[1024]; unsigned grid[3], block[3], shared; size_t nargs; std::vector<std::vector<uint8_t>> args;
    { const char* p = meta_s.c_str(); sscanf(p, "kernel %1023s", kname); p = strstr(p, "grid"); sscanf(p, "grid %u %u %u", grid, grid + 1, grid + 2); p = strstr(p, "block"); sscanf(p, "block %u %u %u", block, block + 1, block + 2); p = strstr(p, "shared"); sscanf(p, "shared %u", &shared); p = strstr(p, "nargs"); sscanf(p, "nargs %zu", &nargs);
      for (size_t i = 0; i < nargs; ++i) { p = strstr(p, "\narg ") + 5; size_t n = strtoul(p, (char**)&p, 10); ++p; std::vector<uint8_t> a(n); for (size_t k = 0; k < n; ++k) { char h[3] = {p[2 * k], p[2 * k + 1], 0}; a[k] = (uint8_t)strtoul(h, 0, 16); } args.push_back(a); } }
    std::vector<Alloc> pre, post; { auto r = slurp(std::string(base) + ".pre", &ok); if (!ok || !parse_mem(r, &pre)) { printf("%06d %s: no .pre (over the capture size cap)\n", seq, kname); ++fail; continue; } auto q = slurp(std::string(base) + ".post", &ok); if (!ok || !parse_mem(q, &post)) { ++fail; continue; } }
    double sim_s = 0; { auto t = slurp(std::string(base) + ".time", &ok); if (ok) sim_s = atof(std::string(t.begin(), t.end()).c_str()); }
    uint64_t bytes = 0; for (auto& a : pre) bytes += a.size;
    Run r1, r2; g_loop_run = &r1; bool g1 = replay(kname, grid, block, shared, args, pre, &r1, ctx); bool g2 = g1 && replay(kname, grid, block, shared, args, pre, &r2, ctx);
    if (!g1 || !g2) { printf("%06d %s: replay failed\n", seq, kname); ++fail; continue; }
    // compare card run 1 against the simulator's post, chunk by chunk; card run 1 against card run 2 for determinism
    uint64_t nchunks = 0, bad = 0, badw = 0, nd_chunks = 0, changed_bytes = 0; float maxulp = 0; std::string first;
    for (size_t i = 0; i < post.size(); ++i) {
      const auto& s = post[i].bytes; const auto& c = r1.res[i].bytes; const auto& c2 = r2.res[i].bytes; const auto& p0 = pre[i].bytes;
      for (uint64_t off = 0; off < s.size(); off += kChunk) { uint64_t n = std::min<uint64_t>(kChunk, s.size() - off); ++nchunks;
        if (memcmp(&p0[off], &s[off], n)) changed_bytes += n;
        if (memcmp(&s[off], &c[off], n)) { ++bad; for (uint64_t k = 0; k + 4 <= n; k += 4) { uint32_t a, b; memcpy(&a, &s[off + k], 4); memcpy(&b, &c[off + k], 4); if (a != b) { ++badw; maxulp = std::max(maxulp, ulps(a, b)); if (first.empty()) { char t[200]; snprintf(t, sizeof t, "first diff at 0x%llx: sim %08x card %08x", (unsigned long long)(post[i].base + off + k), a, b); first = t; } } } }
        if (memcmp(&c[off], &c2[off], n)) ++nd_chunks; }
    }
    const char* verdict = bad == 0 ? "AGREE" : nd_chunks ? "DIFFER (card nondeterministic)" : "DIFFER";
    (bad == 0 ? agree : (nd_chunks ? nondet : differ))++;
    printf("%06d %-40.40s grid %ux%ux%u block %ux%ux%u mem %6.1f MiB changed %6.1f MiB chunks %llu bad %llu words %llu maxulp %.0f | card kernel %9.1f us push %7.2f ms pull %7.2f ms | sim %9.3f ms | %s %s\n", seq, kname, grid[0], grid[1], grid[2], block[0], block[1], block[2], bytes / 1048576.0, changed_bytes / 1048576.0, (unsigned long long)nchunks, (unsigned long long)bad, (unsigned long long)badw, maxulp, r1.kernel_s * 1e6, r1.push_s * 1e3, r1.pull_s * 1e3, sim_s * 1e3, verdict, first.c_str());
    if (r1.loop_sync_s > 0) printf("       resident: launch+sync %.1f us, queued launches %.1f us each (data already on the card; sim %.3f ms -> %.0fx)\n", r1.loop_sync_s * 1e6, r1.loop_queued_s * 1e6, sim_s * 1e3, sim_s / r1.loop_sync_s);
  }
  printf("SUMMARY %s: agree %d differ %d card-nondeterministic %d failed/skipped %d\n", prog.c_str(), agree, differ, nondet, fail);
  return 0;
}
