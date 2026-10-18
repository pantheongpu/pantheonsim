#!/usr/bin/env python3
"""Reduce a rocprofv3 JSON result to the text that does not depend on time,
addresses or the machine, so the trace of a real Radeon card and the trace of
the simulator can be compared with diff.

    rocprof_normalize.py <results.json | directory of rocprofv3 -d output>
                         [--runtime-kernels]
    rocprof_normalize.py --strip-known-gaps < normalised.txt > comparable.txt

What is kept, in the order the profiler numbered the calls (its correlation
ids, which count API calls and dispatches as they happen):

  agent       the GPU the HIP runtime sees: names, CU and wave geometry, ISA id
  kernels     the program's own kernels, as the code object declares them: LDS,
              scratch, kernarg size and register counts
  hip api     every HIP call by thread, with its arguments; a pointer or handle
              prints as <ptr>, a number, enum, dim3 or string as it is
  dispatches  each dispatch of the program's kernels: name, grid and workgroup
              size, group (LDS) and private (scratch) segment bytes
  copies      memory-copy records: direction, bytes, the kinds of agent at each end
  allocations the allocations the program asked for: ALLOCATE records of the
              requested size under hipMalloc, hipHostMalloc, ..., and the FREE
              records under hipFree and its relatives
  markers     ROCTx marks and ranges with their messages, each with the range it was made in
  marker scopes  (only when a HIP call was made inside a ROCTx range) the HIP calls that were,
              with the range: the ancestor of their records

--strip-known-gaps drops from a normalised trace what the simulator does not record yet
(KNOWN_GAPS below, each described in amd/docs/rocprofiler.md). The simulator's trace is
compared with the card's after both have been through it; the card's own trace is compared
whole (run_rocprof_case.sh --card), so a gap that is closed is removed from the list and
the comparison covers it from then on.

What is left out, and why:
  - timestamps, durations, ids, pids and tids, pointers, handles, file paths;
  - HSA API records: ROCm's HIP runtime makes hundreds of HSA calls of its own
    (queue, signal and pool setup) that the simulator's runtime, which reaches
    its devices directly, does not make;
  - the dispatches of the kernels ROCm's runtime runs for the program
    (__amd_rocclr_copyBuffer, ...fillBuffer...: how it copies and fills small
    buffers), unless --runtime-kernels is given;
  - the allocations the runtime makes for itself (staging buffers, pools,
    scratch), which are not the program's;
  - scratch records, whose size is the runtime's pool, not the kernel's.
"""
import glob
import json
import os
import re
import sys

POINTER = re.compile(r"^0x[0-9a-fA-F]+$")

# Allocation APIs: the size argument that the ALLOCATE record is expected to carry.
ALLOCATORS = {
    "hipMalloc": "size", "hipMallocManaged": "size", "hipHostMalloc": "size", "hipExtMallocWithFlags": "sizeBytes",
    "hipMallocAsync": "size", "hipMallocFromPoolAsync": "size", "hipMemAllocHost": "size", "hipHostAlloc": "size",
}
# The buffer-tracing kinds that put records in marker_api (rocprofiler-sdk's rocprofiler_buffer_tracing_kind_t).
MARKER_KINDS = {7: "MARKER_CORE_API", 8: "MARKER_CONTROL_API", 9: "MARKER_NAME_API", 32: "MARKER_CORE_RANGE_API"}
FREERS = {"hipFree", "hipHostFree", "hipFreeAsync", "hipFreeHost", "hipMemFreeHost"}


# What the simulator does not record yet, as it appears in a normalised trace.
#   hip-compiler-api the __hip* calls a hipcc-built program makes (fat binary and function
#                    registration, the launch configuration push and pop)
KNOWN_GAPS = ()


def strip_known_gaps(text):
    out = []
    section = ""
    for line in text.splitlines():
        if line.startswith("["):
            section = line
        elif section.startswith("[hip api"):
            if "hip-compiler-api" in KNOWN_GAPS and line.startswith("__hip"):
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def load(path):
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "**", "*_results.json"), recursive=True))
        if len(found) != 1:
            sys.exit("rocprof_normalize: want exactly one *_results.json under %s, found %d" % (path, len(found)))
        path = found[0]
    with open(path) as f:
        return json.load(f)["rocprofiler-sdk-tool"][0]


def value(arg):
    v = arg["value"]
    return "<ptr>" if POINTER.match(v) else v


def agent_lines(tool):
    keep = ("name", "product_name", "vendor_name", "cu_count", "simd_per_cu", "wave_front_size", "lds_size_in_kb",
            "max_waves_per_cu", "workgroup_max_size", "gfx_target_version", "vendor_id", "device_id")
    out = []
    for a in tool["agents"]:
        if a["type"] != 2 or not a.get("runtime_visibility", {}).get("hip", 1):
            continue
        fields = ["%s=%s" % (k, a[k]) for k in keep if k in a]
        d = a.get("workgroup_max_dim")
        if d:
            fields.append("workgroup_max_dim=%s,%s,%s" % (d["x"], d["y"], d["z"]))
        out.append("gpu " + " ".join(fields))
    return out


def main():
    if "--strip-known-gaps" in sys.argv:
        sys.stdout.write(strip_known_gaps(sys.stdin.read()))
        return
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    runtime_kernels = "--runtime-kernels" in sys.argv
    if len(argv) != 1:
        sys.exit(__doc__)
    tool = load(argv[0])
    buf = tool["buffer_records"]
    names = {k["kind"]: k["operations"] for k in tool["strings"]["buffer_records"]}
    agent_type = {a["id"]["handle"]: {1: "cpu", 2: "gpu"}.get(a["type"], "?") for a in tool["agents"]}
    symbols = {k["kernel_id"]: k for k in tool["kernel_symbols"] if k["kernel_name"]}

    def is_runtime(sym):
        return sym["kernel_name"].startswith("__amd_rocclr_")

    out = []

    out.append("[agent]")
    out += agent_lines(tool)

    out.append("[kernels]")
    for k in sorted((k for k in symbols.values() if runtime_kernels or not is_runtime(k)), key=lambda k: k["kernel_name"]):
        out.append("%s kernarg=%d group=%d private=%d sgpr=%d arch_vgpr=%d accum_vgpr=%d" % (
            k["kernel_name"], k["kernarg_segment_size"], k["group_segment_size"], k["private_segment_size"],
            k["sgpr_count"], k["arch_vgpr_count"], k["accum_vgpr_count"]))

    # The HIP calls by thread, threads numbered in the order they first call.
    hip = sorted(buf["hip_api"], key=lambda r: r["correlation_id"]["internal"])
    by_id = {}
    threads = {}
    api_lines = {}
    hip_scopes = []
    for r in hip:
        kind = "HIP_COMPILER_API" if r["kind"] == 22 else "HIP_RUNTIME_API"
        name = names[kind][r["operation"]]
        args = {a["name"]: value(a) for a in r.get("args", [])}
        by_id[r["correlation_id"]["internal"]] = (name, {a["name"]: a["value"] for a in r.get("args", [])})
        hip_scopes.append((name, r["correlation_id"]["ancestor"]))
        t = threads.setdefault(r["thread_id"], len(threads) + 1)
        api_lines.setdefault(t, []).append("%s(%s)" % (name, ", ".join("%s=%s" % kv for kv in args.items())))
    for t in sorted(api_lines):
        out.append("[hip api, thread %d]" % t)
        out += api_lines[t]

    out.append("[dispatches]")
    for r in sorted(buf["kernel_dispatch"], key=lambda r: r["correlation_id"]["internal"]):
        d = r["dispatch_info"]
        sym = symbols.get(d["kernel_id"])
        if sym is None or (is_runtime(sym) and not runtime_kernels):
            continue
        g, w = d["grid_size"], d["workgroup_size"]
        out.append("%s grid=%d,%d,%d workgroup=%d,%d,%d group=%d private=%d" % (
            sym["demangled_kernel_name"], g["x"], g["y"], g["z"], w["x"], w["y"], w["z"],
            d["group_segment_size"], d["private_segment_size"]))

    out.append("[copies]")
    for r in sorted(buf["memory_copy"], key=lambda r: r["correlation_id"]["internal"]):
        out.append("%s bytes=%d src=%s dst=%s" % (
            names["MEMORY_COPY"][r["operation"]], r["bytes"],
            agent_type.get(r["src_agent_id"]["handle"], "?"), agent_type.get(r["dst_agent_id"]["handle"], "?")))

    out.append("[allocations]")
    for r in sorted(buf["memory_allocation"], key=lambda r: r["correlation_id"]["internal"]):
        op = names["MEMORY_ALLOCATION"][r["operation"]]
        owner = by_id.get(r["correlation_id"]["ancestor"])
        if owner is None:
            continue
        api, args = owner
        if op == "MEMORY_ALLOCATION_ALLOCATE":
            want = ALLOCATORS.get(api)
            if want and args.get(want) == str(r["allocation_size"]):
                out.append("%s %s bytes=%d on=%s" % (api, op, r["allocation_size"], agent_type.get(r["agent_id"]["handle"], "?")))
        elif op == "MEMORY_ALLOCATION_FREE":
            if api in FREERS and int(r["address"], 16) != 0:
                out.append("%s %s" % (api, op))

    out.append("[markers]")
    messages = {m["key"]: m["value"] for m in tool["strings"].get("marker_api", [])}

    def inside(ancestor):
        return " in %r" % messages[ancestor] if ancestor in messages else ""

    for r in sorted(buf["marker_api"], key=lambda r: r["correlation_id"]["internal"]):
        c = r["correlation_id"]
        kind = MARKER_KINDS.get(r["kind"], "MARKER_CORE_RANGE_API")
        text = names[kind][r["operation"]]
        if kind == "MARKER_CORE_RANGE_API":   # the ranges and marks, which have their messages
            text += " %r" % messages.get(c["internal"], "")
        out.append(text + inside(c["ancestor"]))

    scoped = ["%s%s" % (api, inside(a)) for api, a in hip_scopes if a in messages]
    if scoped:
        out.append("[marker scopes]")
        out += scoped

    print("\n".join(out))


if __name__ == "__main__":
    main()
