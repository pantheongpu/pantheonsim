#!/usr/bin/env python3
"""torch.profiler (Kineto) over CUPTI, for a small CUDA workload, checked against the card.

Runs a small program (random matrices, a product, a reduction, a fill, pinned
and pageable copies) under torch.profiler, and compares the sets of names Kineto
collected -- GPU kernels, copies (with their memory kinds), fills, CUDA runtime
and driver calls, synchronizations, user ranges -- with the sets NVIDIA's own
libcupti gave on an RTX 3060 (nvidia/tests/data/pytorch_profiler.card.txt).
Names only: times on the simulator are time spent simulating, and ids and
addresses differ by run.

  profiler_check.py             check against the card's lists; prints "ok <what>"
                                or "FAIL <what>: <why>" per category (run it with
                                `vgpu run --preload`, as run_pytorch.sh does)
  profiler_check.py --card      the same check on a real GPU with NVIDIA's libcupti
  profiler_check.py --print     print the collected names instead of checking
  profiler_check.py --write F   write the collected names to F (how the card's
                                lists are regenerated: on a real GPU, with the
                                torch the file's header names)
  profiler_check.py --json F    also keep Kineto's chrome trace in F

Prints "SKIP: ..." and exits 0 where PyTorch (or a CUDA device) is missing.
"""
import ctypes
import json
import os
import re
import sys
import tempfile

try:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
except Exception as e:  # not installed, or a build that cannot load here
    print(f"SKIP: PyTorch is not usable ({type(e).__name__}: {e})")
    sys.exit(0)
if not torch.version.cuda:
    print("SKIP: this PyTorch is not a CUDA build")
    sys.exit(0)

CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver", "cuda_sync", "user_annotation")
EXPECTED = os.path.join(os.environ.get("VGPU_E2E_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")),
                        "pytorch_profiler.card.txt")

# What the card's lists hold that this simulator's cannot, each with its reason.
# A name that matches is dropped from what was collected before the comparison
# (so the committed lists, which have none of them, are the same on both). Only
# the kernel category has any: no copy, fill or runtime call of the workload
# came from inside a library on the card (checked on the RTX 3060), so there is
# no library-internal copy to allow.
LIBRARY_ONLY = {
    "kernel": [
        # cuBLAS is a host routine on the simulator (nvidia/docs/cublas.md), and no
        # kernel record is made up for a library call (nvidia/docs/cupti.md,
        # "PyTorch's profiler"). NVIDIA's names are chosen by cuBLAS's heuristics
        # (ampere_sgemm_32x32_sliced1x4_nn on the 3060; sm80_xmma_gemm_..., cutlass_...
        # elsewhere): they depend on the library version and the GPU.
        (re.compile(r"^(ampere|volta|turing|hopper|blackwell|sm\d+)_\w*gemm|^cutlass_|^void cutlass::"),
         "a cuBLAS kernel: cuBLAS is a host routine here, with no kernel record"),
    ],
}

# The workload's result is checked against the CPU on the same inputs, to this
# relative tolerance (the product is summed in another order on each side).
RESULT_RTOL = 1e-4


def workload():
    torch.manual_seed(0)
    x = torch.randn(256, 256, device="cuda")
    w = torch.randn(256, 256, device="cuda")
    z = None
    for _ in range(2):
        with record_function("step"):
            y = (x @ w).relu()
            z = y.sum()
            torch.cuda.synchronize()
    host = torch.ones(1024, pin_memory=True)
    dev = host.to("cuda", non_blocking=True)
    back = dev.cpu()
    # A fill by cudaMemsetAsync (PyTorch fills with kernels, so call the runtime
    # it loaded), a pageable copy to the device, and a copy within it.
    zeros = torch.empty(1024, device="cuda")
    cudart = ctypes.CDLL("libcudart.so." + torch.version.cuda.split(".")[0])
    cudart.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
    if cudart.cudaMemsetAsync(zeros.data_ptr(), 0, 4096, torch.cuda.current_stream().cuda_stream) != 0:
        raise RuntimeError("cudaMemsetAsync failed")
    paged = torch.ones(1024).to("cuda")
    twin = (paged + zeros).clone()
    torch.cuda.synchronize()
    return float(z), float(back.sum()) + float(twin.sum()) - 1024.0, x, w


def names_by_category(path):
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    out = {cat: set() for cat in CATEGORIES}
    for e in events:
        cat = e.get("cat")
        if cat in out:
            out[cat].add(e.get("name", ""))
    return out


def drop_library_only(names):
    """Remove the names LIBRARY_ONLY allows to differ; return (kept, dropped)."""
    kept, dropped = {}, {}
    for cat, ns in names.items():
        kept[cat], dropped[cat] = set(), {}
        for n in ns:
            why = next((w for rx, w in LIBRARY_ONLY.get(cat, ()) if rx.search(n)), None)
            if why:
                dropped[cat][n] = why
            else:
                kept[cat].add(n)
    return kept, dropped


def render(names, header=()):
    lines = [f"# {h}" for h in header]
    for cat in CATEGORIES:
        lines.append(f"[{cat}]")
        lines += ["  " + n for n in sorted(names.get(cat, ()))]
    return "\n".join(lines) + "\n"


def read_expected(path):
    out, cat, header = {}, None, []
    with open(path) as f:
        for line in f.read().splitlines():
            if line.startswith("#"):
                header.append(line[1:].strip())
            elif line.startswith("[") and line.endswith("]"):
                cat = line[1:-1]
                out[cat] = set()
            elif line.startswith("  ") and cat is not None:
                out[cat].add(line[2:])
    return out, header


def check(names, result):
    expected, header = read_expected(EXPECTED)
    kept, dropped = drop_library_only(names)
    bad = 0
    # The random numbers PyTorch draws depend on the GPU (its launch shape follows the
    # multiprocessor count), so the result is checked against the CPU on the same inputs,
    # not against the card's number.
    z, back_sum, x, w = result
    ref = float((x.cpu().double() @ w.cpu().double()).relu().sum())
    if abs(z - ref) <= RESULT_RTOL * abs(ref) and back_sum == 1024.0:
        print("ok workload result (matches the CPU's)")
    else:
        print(f"FAIL workload result: {z}, {back_sum}; the CPU's {ref}, 1024.0")
        bad += 1
    for cat in CATEGORIES:
        missing = sorted(expected.get(cat, set()) - kept[cat])
        extra = sorted(kept[cat] - expected.get(cat, set()))
        if not missing and not extra:
            note = f"; {len(dropped[cat])} more allowed to differ" if dropped[cat] else ""
            print(f"ok profiler {cat} names ({len(kept[cat])} as on the card{note})")
            continue
        bad += 1
        # run_pytorch.sh shows only the "ok"/"FAIL" lines, so the names go on this one.
        short = lambda ns: "[" + "; ".join(n[:90] for n in ns) + "]"
        print(f"FAIL profiler {cat} names: the card's profiler lists {short(missing)} and this one did not;"
              f" this one lists {short(extra)} and the card's did not")
        for n in missing:
            print("      missing: " + n[:300])
        for n in extra:
            print("      extra:   " + n[:300])
    if bad:
        have = next((h for h in header if h.startswith("torch ")), "?")
        print(f"      (the card's lists are from {have}; this run is torch {torch.__version__}."
              " Regenerate with --write on a real GPU if PyTorch's kernel names changed)")
    return bad


def main():
    args = sys.argv[1:]
    keep = write = None
    mode = "check"
    while args:
        a = args.pop(0)
        if a == "--json":
            keep = args.pop(0)
        elif a == "--write":
            write = args.pop(0)
        elif a == "--print":
            mode = "print"
        elif a == "--card":
            pass  # the check is the same; this says the process is on a real GPU
        else:
            print(f"unknown argument {a}", file=sys.stderr)
            return 2
    if not torch.cuda.is_available():
        print("SKIP: PyTorch sees no CUDA device")
        return 0
    # PyTorch reaches the driver API through the runtime's initialisation; the
    # simulator's driver shim wants cuInit first, as a driver does of a program
    # that never called the runtime (idempotent, so harmless against NVIDIA's).
    ctypes.CDLL("libcuda.so.1").cuInit(0)
    workload()  # warm-up: libraries load and kernels are first used outside the profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = workload()
    path = keep or os.path.join(tempfile.mkdtemp(), "trace.json")
    prof.export_chrome_trace(path)
    names = names_by_category(path)
    if write:
        kept, _ = drop_library_only(names)
        gpu = torch.cuda.get_device_name(0)
        with open(write, "w") as f:
            f.write(render(kept, [
                f"torch {torch.__version__}, CUDA {torch.version.cuda}, on {gpu}, NVIDIA's libcupti",
                "Names Kineto collected for nvidia/tests/pytorch/profiler_check.py's workload.",
                "cuBLAS's own kernels are left out (LIBRARY_ONLY in the script).",
                "Regenerate on a real GPU: profiler_check.py --write <this file>",
            ]))
        print("wrote", write)
        return 0
    if mode == "print":
        print("result:", result[:2])
        print(render(names), end="")
        return 0
    return 1 if check(names, result) else 0


if __name__ == "__main__":
    sys.exit(main())
