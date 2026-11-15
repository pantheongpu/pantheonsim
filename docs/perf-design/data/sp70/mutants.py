#!/usr/bin/env python3
"""Round sp70: mutants of the three changes. Each is one edit of a source file of a build tree (put back afterwards); the test named
with it is rebuilt and must fail.

  mutants.py --src <tree> --build <tree>/build [--only N] [--lock /tmp/pantheonsim-builder.lock]

A test is a ctest regular expression, run with `ctest -R <re> -j8` in the build directory after `cmake --build --target <targets>`.
"""
import argparse
import os
import subprocess

EX = "amd/src/gcn_exec.cpp"
TI = "amd/src/gcn_tier.cpp"
KI = TI   # (the fmin/fmax kernels are at the end of the tier)
HC = "include/vgpu/host_cores.hpp"
POOL = ("test_amd_gcn_register_pool", "^test_amd_gcn_register_pool$")
THREADS = ("test_amd_gcn_threads", "^test_amd_gcn_threads")
CORES = ("test_host_cores", "^host_cores$")
BITS = ("test_amd_gcn_tier_same_bits", "^test_amd_gcn_tier_same_bits_shard")
RUNS = ("test_amd_gcn_memruns", "^test_amd_gcn_memruns")
M = [
    (EX, POOL, "pool: room for 32 files, as before", "      static constexpr size_t kMax = 288;", "      static constexpr size_t kMax = 32;"),
    (EX, POOL, "pool: room for 128 files", "      static constexpr size_t kMax = 288;", "      static constexpr size_t kMax = 128;"),
    (EX, POOL, "pool: nothing is given back", "      if (!pl.alive || pl.count == Pool::kMax) return false;", "      if (true) return false;"),
    (EX, THREADS, "threads: the hardware threads, as before", "    want = vgpu::host_cores();", "    want = vgpu::host_cpus();"),
    (EX, THREADS, "threads: VGPU_THREADS capped by the cores", "    want = v > 0 ? static_cast<unsigned>(v) : 1;",
     "    want = v > 0 ? std::min(static_cast<unsigned>(v), vgpu::host_cores()) : 1;"),
    (HC, CORES, "cores: every CPU its own core", "    cores.emplace(package, core);", "    cores.emplace(package, cpu);"),
    (HC, CORES, "cores: the package not looked at", "    cores.emplace(package, core);", "    cores.emplace(0, core);"),
    (HC, CORES, "cores: a CPU without a topology is skipped", "    if (!(fp >> package) || !(fc >> core)) return 0;",
     "    if (!(fp >> package) || !(fc >> core)) continue;"),
    (KI, BITS, "fmax: b a NaN gives b", "    r = _mm256_blendv_pd(r, a, bnan);\n", ""),
    (KI, BITS, "fmax: a signaling NaN is not answered with a NaN", "    r = _mm256_blendv_pd(r, _mm256_add_pd(a, b), snan);\n", ""),
    (KI, BITS, "fmax: the operands the other way round", "    __m256d r = Max ? _mm256_max_pd(a, b) : _mm256_min_pd(a, b);",
     "    __m256d r = Max ? _mm256_max_pd(b, a) : _mm256_min_pd(b, a);"),
    (KI, BITS, "fmax: minimum for maximum", "    __m256d r = Max ? _mm256_max_pd(a, b) : _mm256_min_pd(a, b);",
     "    __m256d r = Max ? _mm256_min_pd(a, b) : _mm256_max_pd(a, b);"),
    (KI, BITS, "fmax: only a's signaling NaN", "    const __m256d snan = _mm256_or_pd(_mm256_and_pd(anan, aq), _mm256_and_pd(bnan, bq));",
     "    const __m256d snan = _mm256_and_pd(anan, aq);"),
    (EX, RUNS, "host batch: a quad's loads not made", "    return mem.host_map_possible(addrs[0]) ? mem.host_load_batch(addrs, n, bytes, vals) || mem.load_batch(addrs, n, bytes, vals)\n",
     "    return mem.host_map_possible(addrs[0]) ? true\n"),
    (EX, RUNS, "host batch: a quad's stores not made", "  return store_wave(a, n, 4, v);", "  return mem.host_map_possible(a[0]) ? true : mem.store_batch(a, n, 4, v);"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    killed = total = 0
    for i, (f, (target, regex), name, old, new) in enumerate(M):
        if only and i not in only:
            continue
        total += 1
        path = os.path.join(a.src, f)
        text = open(path).read()
        if text.count(old) != 1:
            print("%2d NOT APPLIED (%d matches): %s" % (i, text.count(old), name), flush=True)
            continue
        try:
            open(path, "w").write(text.replace(old, new))
            b = subprocess.run(["flock", a.lock, "nice", "-n", "19", "cmake", "--build", a.build, "--target", target, "-j3"],
                               capture_output=True, text=True)
            if b.returncode != 0:
                print("%2d DID NOT BUILD: %s\n%s" % (i, name, b.stdout[-600:]), flush=True)
                continue
            t = subprocess.run(["ctest", "-R", regex, "-j8", "--timeout", "1800"], cwd=a.build, capture_output=True, text=True)
            failed = [l.split(" - ")[1].split(" (")[0] for l in t.stdout.splitlines() if " - " in l and "(Failed)" in l or " - " in l and "(Timeout)" in l]
            if t.returncode != 0 and failed:
                killed += 1
                print("%2d killed by %s: %s" % (i, ", ".join(failed[:4]) + (" ..." if len(failed) > 4 else ""), name), flush=True)
            else:
                print("%2d SURVIVED (%s): %s" % (i, regex, name), flush=True)
        finally:
            open(path, "w").write(text)
    print("%d of %d killed" % (killed, total), flush=True)


if __name__ == "__main__":
    main()
