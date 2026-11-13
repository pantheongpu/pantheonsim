#!/usr/bin/env python3
"""Round sp60b: mutants of the sampling rule (include/vgpu/host_gpu.hpp). Each is a copy of the header with one change, put in a directory outside
the tree that comes first on the include path; tests/unit/test_host_gpu_logic.cpp is compiled against it and must fail.

  mutants.py --src <tree> --work <dir outside the tree> [--only N] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse
import fcntl
import os
import subprocess
import time

M = [
    ("sample_blocks: a grid of exactly the minimum is not sampled", "if (blocks < kSampleMinBlocks) return 0;", "if (blocks <= kSampleMinBlocks) return 0;"),
    ("sample_blocks: no cap", "return std::min<uint64_t>(blocks / 2, kSampleMaxBlocks);", "return blocks / 2;"),
    ("sample_blocks: the whole grid", "return std::min<uint64_t>(blocks / 2, kSampleMaxBlocks);", "return std::min<uint64_t>(blocks, 8 * kSampleMaxBlocks);"),
    ("sample_blocks: no minimum grid", "if (blocks < kSampleMinBlocks) return 0;\n  return std::min", "return std::min"),
    ("affordable: what was spent does not count", "return spent_s + std::max(expected_s, 0.0) <= kSampleShareOfAge * age_s;", "return std::max(expected_s, 0.0) <= kSampleShareOfAge * age_s;"),
    ("affordable: what is expected does not count", "return spent_s + std::max(expected_s, 0.0) <= kSampleShareOfAge * age_s;", "return spent_s <= kSampleShareOfAge * age_s;"),
    ("affordable: a negative expectation is a credit", "return spent_s + std::max(expected_s, 0.0) <= kSampleShareOfAge * age_s;", "return spent_s + expected_s <= kSampleShareOfAge * age_s;"),
    ("affordable: the whole age, not a share", "return spent_s + std::max(expected_s, 0.0) <= kSampleShareOfAge * age_s;", "return spent_s + std::max(expected_s, 0.0) <= age_s;"),
    ("affordable: always", "return spent_s + std::max(expected_s, 0.0) <= kSampleShareOfAge * age_s;", "return true;"),
    ("estimate: a sample that timed nothing extrapolates", "if (s.timed == 0 || s.total == 0 || !(s.seconds > 0)) return std::nullopt;", "if (s.total == 0 || !(s.seconds > 0)) return std::nullopt;"),
    ("estimate: no time is a sample", "if (s.timed == 0 || s.total == 0 || !(s.seconds > 0)) return std::nullopt;", "if (s.timed == 0 || s.total == 0) return std::nullopt;"),
    ("estimate: no grid is a sample", "if (s.timed == 0 || s.total == 0 || !(s.seconds > 0)) return std::nullopt;", "if (s.timed == 0 || !(s.seconds > 0)) return std::nullopt;"),
    ("estimate: not divided by the threads", "return per_block * static_cast<double>(s.total) / static_cast<double>(std::max<unsigned>(s.threads, 1));", "return per_block * static_cast<double>(s.total);"),
    ("estimate: zero threads divide by zero", "/ static_cast<double>(std::max<unsigned>(s.threads, 1));", "/ static_cast<double>(s.threads);"),
    ("stride: the first blocks alone", "return n ? blocks / n : 1;", "return 1;"),
    ("stride: past the end of the grid", "return n ? blocks / n : 1;", "return n ? blocks / n + 1 : 1;"),
    ("stride: half the grid", "return n ? blocks / n : 1;", "return n ? std::max<uint64_t>(blocks / n / 2, 2) : 1;"),
    ("stride: a division by zero for a grid that is not sampled", "return n ? blocks / n : 1;", "return blocks / n;"),
    ("estimate: one block more than were timed", "const double per_block = s.seconds / static_cast<double>(s.timed);", "const double per_block = s.seconds / static_cast<double>(s.timed + 1);"),
    ("estimate: the timed blocks taken off the grid", "return per_block * static_cast<double>(s.total) /", "return per_block * static_cast<double>(s.total - s.timed) /"),
    ("path: nothing pushed", "e.push_bytes = push_bytes;\n  e.pull_bytes = live_bytes;", "e.push_bytes = 0;\n  e.pull_bytes = live_bytes;"),
    ("path: nothing charged for coming back", "e.push_bytes = push_bytes;\n  e.pull_bytes = live_bytes;", "e.push_bytes = push_bytes;\n  e.pull_bytes = 0;"),
    ("path: the live memory pushed whatever the card holds", "e.push_bytes = push_bytes;\n  e.pull_bytes = live_bytes;", "e.push_bytes = live_bytes;\n  e.pull_bytes = live_bytes;"),
    ("path: the card's kernel time is the default whatever the launch", "std::max(c.kernel_s_default, e.sim_s / kSampleCardSpeedup);", "c.kernel_s_default;"),
    ("path: the card's kernel time may be under the default", "std::max(c.kernel_s_default, e.sim_s / kSampleCardSpeedup);", "e.sim_s / kSampleCardSpeedup;"),
    ("path: a measured card time is ignored", "e.card_kernel_s = card_kernel_s ? *card_kernel_s : std::max(c.kernel_s_default, e.sim_s / kSampleCardSpeedup);", "e.card_kernel_s = std::max(c.kernel_s_default, e.sim_s / kSampleCardSpeedup);"),
    ("path: the start is charged with the driver up", "e.start_s = driver_started ? 0 : std::max(c.start_cost_s, 0.0);\n  return e;\n}\n\n// Whether a sampled", "e.start_s = std::max(c.start_cost_s, 0.0);\n  return e;\n}\n\n// Whether a sampled"),
    ("path: the start is never charged", "e.start_s = driver_started ? 0 : std::max(c.start_cost_s, 0.0);\n  return e;\n}\n\n// Whether a sampled", "e.start_s = 0;\n  return e;\n}\n\n// Whether a sampled"),
    ("path: a negative start cost is a credit", "e.start_s = driver_started ? 0 : std::max(c.start_cost_s, 0.0);\n  return e;\n}\n\n// Whether a sampled", "e.start_s = driver_started ? 0 : c.start_cost_s;\n  return e;\n}\n\n// Whether a sampled"),
    ("offload: any sample with an estimate goes", "return should_offload(sample_estimate(s, push_bytes, live_bytes, c, driver_started, card_kernel_s), c);", "return true;"),
    ("offload: the start is dropped from the decision", "return should_offload(sample_estimate(s, push_bytes, live_bytes, c, driver_started, card_kernel_s), c);", "return should_offload(sample_estimate(s, push_bytes, live_bytes, c, true, card_kernel_s), c);"),
    ("offload: the measured card time is dropped from the decision", "return should_offload(sample_estimate(s, push_bytes, live_bytes, c, driver_started, card_kernel_s), c);", "return should_offload(sample_estimate(s, push_bytes, live_bytes, c, driver_started), c);"),
    ("worth: a grid with nothing to sample is sampled", "if (!w.loop || sample_blocks(w.blocks) == 0) return false;", "if (!w.loop) return false;"),
    ("worth: a short whole launch does not stop the sampling", "if (w.whole_s && *w.whole_s <= c.margin * card_path_s) return false;\n", ""),
    ("worth: a whole launch at the bar is sampled", "if (w.whole_s && *w.whole_s <= c.margin * card_path_s) return false;", "if (w.whole_s && *w.whole_s < c.margin * card_path_s) return false;"),
    ("worth: the bar has no margin", "if (w.whole_s && *w.whole_s <= c.margin * card_path_s) return false;", "if (w.whole_s && *w.whole_s <= card_path_s) return false;"),
    ("worth: neither the floor nor the age is asked", "return w.static_s >= sample_static_floor(w.live_bytes) || sample_affordable(w.age_s, w.spent_s, w.expected_s);", "return true;"),
    ("worth: the floor alone", "return w.static_s >= sample_static_floor(w.live_bytes) || sample_affordable(w.age_s, w.spent_s, w.expected_s);", "return w.static_s >= sample_static_floor(w.live_bytes);"),
    ("worth: the age alone", "return w.static_s >= sample_static_floor(w.live_bytes) || sample_affordable(w.age_s, w.spent_s, w.expected_s);", "return sample_affordable(w.age_s, w.spent_s, w.expected_s);"),
    ("worth: the floor excludes its own value", "return w.static_s >= sample_static_floor(w.live_bytes) ||", "return w.static_s > sample_static_floor(w.live_bytes) ||"),
    ("worth: the floor and the age", "return w.static_s >= sample_static_floor(w.live_bytes) || sample_affordable(w.age_s, w.spent_s, w.expected_s);", "return w.static_s >= sample_static_floor(w.live_bytes) && sample_affordable(w.age_s, w.spent_s, w.expected_s);"),
    ("worth: what the samples cost so far is not asked", "sample_affordable(w.age_s, w.spent_s, w.expected_s);\n}", "sample_affordable(w.age_s, 0, w.expected_s);\n}"),
    ("worth: the expected cost is the guess whatever was seen", "sample_affordable(w.age_s, w.spent_s, w.expected_s);\n}", "sample_affordable(w.age_s, w.spent_s, kSampleCostGuessSeconds);\n}"),
    ("peek: not scaled by the grid", "if (it == table_.end() || !it->second.samples) return std::nullopt;\n    return it->second.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));", "if (it == table_.end() || !it->second.samples) return std::nullopt;\n    return it->second.per_block_s;"),
    ("peek: needs a full history", "if (it == table_.end() || !it->second.samples) return std::nullopt;\n    return it->second.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));", "if (it == table_.end() || it->second.samples < kSamplesNeeded) return std::nullopt;\n    return it->second.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));"),
    ("record_extrapolated: one sample like any other", "    e.samples = std::max(e.samples, kSamplesNeeded);\n", ""),
    ("record_extrapolated: records nothing", "    record(kernel, blocks, seconds);\n    Entry& e = table_[kernel];\n    e.samples = std::max(e.samples, kSamplesNeeded);", "    Entry& e = table_[kernel];\n    e.samples = std::max(e.samples, kSamplesNeeded);"),
    ("floor: the live memory does not raise it", "return std::max(kSampleStaticFloorSeconds, 3.0 * kSampleSecondsPerLiveByte * static_cast<double>(live_bytes));", "return kSampleStaticFloorSeconds;"),
    ("floor: the faults' cost alone, not three times it", "return std::max(kSampleStaticFloorSeconds, 3.0 * kSampleSecondsPerLiveByte * static_cast<double>(live_bytes));", "return std::max(kSampleStaticFloorSeconds, kSampleSecondsPerLiveByte * static_cast<double>(live_bytes));"),
    ("floor: the two added", "return std::max(kSampleStaticFloorSeconds, 3.0 * kSampleSecondsPerLiveByte * static_cast<double>(live_bytes));", "return kSampleStaticFloorSeconds + 3.0 * kSampleSecondsPerLiveByte * static_cast<double>(live_bytes);"),
    ("constant: the faults' cost per byte", "inline constexpr double kSampleSecondsPerLiveByte = 6e-10;", "inline constexpr double kSampleSecondsPerLiveByte = 6e-12;"),
    ("worth: a kernel with no loop is sampled", "if (!w.loop || sample_blocks(w.blocks) == 0) return false;", "if (sample_blocks(w.blocks) == 0) return false;"),
    ("loop: a cycle the entry cannot reach counts", "for (size_t root = 0; root < std::min<size_t>(n, 1); ++root) {\n    stack.emplace_back(root, 0);", "for (size_t root = 0; root < n; ++root) {\n    if (state[root]) continue;\n    stack.emplace_back(root, 0);"),
    ("loop: an instruction seen before counts as a cycle", "if (state[next] == 1) return true;", "if (state[next] != 0) return true;"),
    ("loop: no cycle is ever found", "if (state[next] == 1) return true;", ""),
    ("loop: an exit goes on", "if (edge == 0 && code[i].falls_through && i + 1 < n) next = i + 1;", "if (edge == 0 && i + 1 < n) next = i + 1;"),
    ("loop: a branch is never taken", "if (edge == 1 && code[i].target >= 0 && static_cast<size_t>(code[i].target) < n) next = static_cast<size_t>(code[i].target);", ""),
    ("loop: indirect flow is ignored", "  for (const FlowInsn& f : code)\n    if (f.indirect) return true;\n", ""),
    ("timeout: the live memory does not lengthen it", "return kSampleTimeoutSeconds + kSampleTimeoutSecondsPerLiveByte * static_cast<double>(live_bytes);", "return kSampleTimeoutSeconds;"),
    ("timeout: the live memory alone", "return kSampleTimeoutSeconds + kSampleTimeoutSecondsPerLiveByte * static_cast<double>(live_bytes);", "return kSampleTimeoutSecondsPerLiveByte * static_cast<double>(live_bytes);"),
    ("constant: the timeout per byte", "inline constexpr double kSampleTimeoutSecondsPerLiveByte = 2e-10;", "inline constexpr double kSampleTimeoutSecondsPerLiveByte = 2e-8;"),
    ("constant: the static floor", "inline constexpr double kSampleStaticFloorSeconds = 0.030;", "inline constexpr double kSampleStaticFloorSeconds = 0.001;"),
    ("constant: the card's assumed speed-up", "inline constexpr double kSampleCardSpeedup = 8.0;", "inline constexpr double kSampleCardSpeedup = 2000.0;"),
    ("constant: the most blocks", "inline constexpr uint64_t kSampleMaxBlocks = 32;", "inline constexpr uint64_t kSampleMaxBlocks = 4096;"),
    ("constant: the smallest grid", "inline constexpr uint64_t kSampleMinBlocks = 256;", "inline constexpr uint64_t kSampleMinBlocks = 8;"),
    ("constant: the budget", "inline constexpr double kSampleBudgetSeconds = 0.002;", "inline constexpr double kSampleBudgetSeconds = 1.0;"),
    ("constant: the timeout", "inline constexpr double kSampleTimeoutSeconds = 0.050;", "inline constexpr double kSampleTimeoutSeconds = 5.0;"),
    ("constant: the share of the age", "inline constexpr double kSampleShareOfAge = 0.02;", "inline constexpr double kSampleShareOfAge = 0.5;"),
    ("constant: the first sample's guess", "inline constexpr double kSampleCostGuessSeconds = 0.003;", "inline constexpr double kSampleCostGuessSeconds = 0.0;"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--hold", type=float, default=200)
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    work = os.path.abspath(a.work)
    assert not work.startswith(src + os.sep), "the mutants go outside the tree"
    os.makedirs(os.path.join(work, "inc/vgpu"), exist_ok=True)
    header = open(os.path.join(src, "include/vgpu/host_gpu.hpp")).read()
    pre = ["nice", "-n", "19"]
    fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666) if a.lock else None
    since = None
    killed = 0
    todo = [(i, m) for i, m in enumerate(M) if a.only < 0 or i == a.only]
    for i, (name, old, new) in todo:
        if header.count(old) != 1:
            print("%2d NOT APPLIED (%d matches): %s" % (i, header.count(old), name), flush=True)
            continue
        if fd is not None and since is not None and time.monotonic() - since > a.hold:   # the machine's build lock: given up now and then
            fcntl.flock(fd, fcntl.LOCK_UN)
            since = None
            time.sleep(1)
        if fd is not None and since is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            since = time.monotonic()
        open(os.path.join(work, "inc/vgpu/host_gpu.hpp"), "w").write(header.replace(old, new))
        exe = os.path.join(work, "mut%d" % i)
        c = subprocess.run(pre + ["g++", "-std=c++20", "-O1", "-I", os.path.join(work, "inc"), "-I", os.path.join(src, "include"), "-I", os.path.join(src, "tests/framework"),
                                  os.path.join(src, "tests/unit/test_host_gpu_logic.cpp"), "-o", exe], capture_output=True, text=True)
        if c.returncode != 0:
            print("%2d killed (does not compile): %s" % (i, name), flush=True)
            killed += 1
            continue
        r = subprocess.run([exe], capture_output=True, text=True)
        fails = [l.split("]", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("[ FAIL ]")]
        os.remove(exe)
        if r.returncode != 0:
            killed += 1
            print("%2d killed by %s: %s" % (i, ", ".join(fails[:3]) or "exit %d" % r.returncode, name), flush=True)
        else:
            print("%2d SURVIVED: %s" % (i, name), flush=True)
    print("%d of %d killed" % (killed, len(todo)))


if __name__ == "__main__":
    main()
