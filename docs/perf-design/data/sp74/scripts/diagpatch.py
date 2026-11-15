#!/usr/bin/env python3
"""diagpatch.py <tree>: a diagnostic build (never committed): SP74_DIAG=<file> writes, per tier instruction form, calls, lanes, hand-backs and TSC ticks in its handler."""
import sys
p = sys.argv[1] + "/amd/src/gcn_exec.cpp"
s = open(p).read()
a = "struct Machine {\n"
assert s.count(a) == 1
diag = r'''
#include <dlfcn.h>
#include <x86intrin.h>
#include <unordered_map>
namespace sp74diag {
struct E { uint64_t n = 0, ticks = 0, lanes = 0, bails = 0; std::string what; };
std::mutex g_m;
std::map<std::string, E> g_all;
struct T;
std::set<T*> g_live;
void fold(std::unordered_map<uint64_t, E>& m) {
  for (auto& [k, e] : m) { E& x = g_all[e.what]; x.n += e.n, x.ticks += e.ticks, x.lanes += e.lanes, x.bails += e.bails; }
  m.clear();
}
struct T {
  std::unordered_map<uint64_t, E> m;
  T() { std::lock_guard<std::mutex> l(g_m); g_live.insert(this); }
  ~T() { std::lock_guard<std::mutex> l(g_m); fold(m); g_live.erase(this); }
};
const char* path() { static const char* p = std::getenv("SP74_DIAG"); return p; }
struct Dump {
  ~Dump() {
    if (!path()) return;
    std::lock_guard<std::mutex> l(g_m);
    for (T* t : g_live) fold(t->m);
    if (FILE* f = std::fopen(path(), "a")) {
      for (auto& [w, e] : g_all) std::fprintf(f, "%llu %llu %llu %llu %s\n", (unsigned long long)e.ticks, (unsigned long long)e.n, (unsigned long long)e.lanes, (unsigned long long)e.bails, w.c_str());
      std::fclose(f);
    }
  }
} g_dump;
inline E& slot(const gcn::Inst& in) {
  static thread_local T t;
  const gcn::tier::Op& o = in.fast;
  uint64_t k = reinterpret_cast<uint64_t>(o.run) * 0x9E3779B97F4A7C15ull ^ reinterpret_cast<uint64_t>(o.kern) * 0xC2B2AE3D27D4EB4Full ^ in.opcode * 0x165667B19E3779F9ull;
  for (int i = 0; i < 3; ++i) k = k * 31 + (o.src[i].kind | o.src[i].wide << 4 | o.src[i].neg << 5 | o.src[i].abs << 6 | o.src[i].sel << 8);
  k = k * 31 + (o.omod | o.clamp << 4 | o.sub << 8 | static_cast<uint64_t>(o.dsel) << 40 | static_cast<uint64_t>(static_cast<int>(in.enc)) << 48);
  E& e = t.m[k];
  if (e.what.empty()) {
    std::string w = in.name + " enc" + std::to_string(static_cast<int>(in.enc));
    for (int i = 0; i < 3; ++i) {
      const auto& s = o.src[i];
      if (!s.kind) break;
      w += ' ';
      if (s.neg) w += '-';
      if (s.abs) w += '|';
      w += s.kind == gcn::tier::kVgpr ? 'v' : s.kind == gcn::tier::kConst ? 'c' : s.kind == gcn::tier::kSgpr ? 's' : '?';
      if (s.wide) w += 'w';
      if (s.sel != 6) w += "sel" + std::to_string(s.sel);
    }
    Dl_info di{};
    dladdr(reinterpret_cast<void*>(o.run), &di);
    char b[96];
    std::snprintf(b, sizeof b, " omod%d clamp%d sub%u run@%lx kern@%lx", o.omod, o.clamp, o.sub, reinterpret_cast<unsigned long>(o.run) - reinterpret_cast<unsigned long>(di.dli_fbase), o.kern ? reinterpret_cast<unsigned long>(o.kern) - reinterpret_cast<unsigned long>(di.dli_fbase) : 0ul);
    e.what = w + b;
  }
  return e;
}
}  // namespace sp74diag
'''
s = s.replace(a, diag + a)
b = "    const int done = in.fast.run(in.fast, r);\n    if (done == 3) {"
assert s.count(b) == 1
s = s.replace(b, '''    int done;
    if (__builtin_expect(sp74diag::path() != nullptr, 0)) {
      const uint64_t ex0 = w.exec;
      const uint64_t t0 = __rdtsc();
      done = in.fast.run(in.fast, r);
      const uint64_t t1 = __rdtsc();
      sp74diag::E& e = sp74diag::slot(in);
      e.ticks += t1 - t0, e.n += 1, e.lanes += static_cast<uint64_t>(__builtin_popcountll(ex0)), e.bails += done == 0;
    } else {
      done = in.fast.run(in.fast, r);
    }
    if (done == 3) {''')
open(p, "w").write(s)
