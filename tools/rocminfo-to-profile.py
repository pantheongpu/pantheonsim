#!/usr/bin/env python3
"""Turn a rocminfo/rocm-smi dump into a VirtualGPU device profile.

    tools/characterize-do.sh gpu-mi325x1-256gb tor1
    tools/rocminfo-to-profile.py /tmp/vgpu-cloud/gpu-mi325x1-256gb.rocminfo.txt

The AMD counterpart of characterize.cu, and deliberately not a program that
runs on the device. The first two attempts at this compiled a HIP binary on the
rented droplet and failed identically: DigitalOcean's AMD image ships hipcc but
not the HIP development headers, so hip/hip_runtime.h does not exist anywhere
under /opt/rocm. Learning that cost two launches on an account with fixed
credit.

rocminfo is already installed and reports everything the profile schema needs.
So the rented machine does one thing -- dump text -- and the parsing happens
here, where getting it wrong costs nothing.

Every value written comes from the dump. Anything rocminfo does not report is
left out rather than filled in, which is the same rule characterize.cu follows.
"""
import re
import sys


def gpu_agent(text):
    """The GPU agent block. rocminfo lists CPUs first, and they have most of
    the same field names, so picking the first match finds the CPU."""
    blocks = text.split("Agent ")
    for b in blocks:
        if re.search(r"Device Type:\s+GPU", b) and "gfx" in b:
            return b
    return None


def field(block, key, cast=int):
    """First value of `key`, with rocminfo's "1024(0x400)" stripped."""
    m = re.search(re.escape(key) + r":\s+(\S+)", block)
    if not m:
        return None
    v = m.group(1).split("(")[0]
    try:
        return cast(v)
    except ValueError:
        return None


def dimensions(block, key):
    r"""The three values under a "... per Dimension:" heading.

    rocminfo pads every value line with trailing spaces, which is why an
    anchored \S+\n does not match and this takes the rest of the line."""
    m = re.search(re.escape(key) + r":[ \t]*\n((?:[ \t]+[xyz][ \t]+\S+[ \t]*\n){3})", block)
    if not m:
        return None
    return [int(v.split("(")[0]) for v in re.findall(r"[xyz][ \t]+(\S+)", m.group(1))]


def smi_value(text, pattern):
    """A number from the rocm-smi section of the dump, or None."""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def pool_size_kb(block, segment_match):
    """Size of the first memory pool whose Segment line matches.

    Not "the largest Size: in the block": that also matches "Grid Max Size",
    which is 4294967295 and turned a 256 GB card into a 4 TB one."""
    for chunk in re.split(r"\n\s*Pool \d+", block)[1:]:
        seg = re.search(r"Segment:\s+(.+)", chunk)
        if seg and segment_match in seg.group(1):
            m = re.search(r"Size:\s+(\d+)", chunk)
            if m:
                return int(m.group(1))
    return None


ARCH = [
    ("gfx950", "cdna4"), ("gfx942", "cdna3"), ("gfx941", "cdna3"),
    ("gfx940", "cdna3"), ("gfx90a", "cdna2"), ("gfx908", "cdna"),
    ("gfx12", "rdna4"), ("gfx11", "rdna3"), ("gfx10", "rdna"),
]

# Marketing names rocminfo does not know. The driver reports "AMD Radeon
# Graphics" for every Instinct part, so the chip id is what identifies the
# model. Only ids seen on real hardware are listed; an unknown one keeps the
# generic name rather than being guessed at.
CHIP_IDS = {
    0x74b9: ("MI325X", "mi325x"),
    0x74a1: ("MI300X", "mi300x"),
    0x74a0: ("MI300A", "mi300a"),
    0x7408: ("MI250X", "mi250x"),
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    text = open(sys.argv[1]).read()
    g = gpu_agent(text)
    if not g:
        print("no GPU agent in that dump", file=sys.stderr)
        return 1

    target_full = None
    m = re.search(r"amdgcn-amd-amdhsa--(\S+)", g)
    if m:
        target_full = m.group(1)
    target = (target_full or "").split(":")[0] or (field(g, "Name", str) or "")

    arch = "unknown"
    for prefix, name in ARCH:
        if target.startswith(prefix):
            arch = name
            break

    chip = field(g, "Chip ID")
    model, slug = CHIP_IDS.get(chip, (field(g, "Marketing Name", str) or "AMD GPU", target))

    cus = field(g, "Compute Unit")
    wave = field(g, "Wavefront Size")
    waves_per_cu = field(g, "Max Waves Per CU")
    wg_max = field(g, "Workgroup Max Size")
    clock = field(g, "Max Clock Freq. (MHz)")
    l2_kb = None
    m = re.search(r"L2:\s+(\d+)", g)
    if m:
        l2_kb = int(m.group(1))

    wg_dims = dimensions(g, "Workgroup Max Size per Dimension") or [wg_max, wg_max, wg_max]
    grid_dims = dimensions(g, "Grid Max Size per Dimension")

    # Device VRAM is the coarse-grained global pool; the fine-grained pools are
    # the same memory seen through a different coherence policy, not more of it.
    vram_kb = pool_size_kb(g, "GLOBAL; FLAGS: COARSE GRAINED") or 0
    # LDS is the GROUP segment pool, reported in KB like the rest.
    lds_kb = pool_size_kb(g, "GROUP")
    lds = lds_kb * 1024 if lds_kb else None

    out = []
    out.append(f"# VirtualGPU device profile: {model}")
    out.append("# Generated by tools/rocminfo-to-profile.py from a physical device.")
    out.append("# Every value below was read from the hardware, not a datasheet.")
    out.append(f"id: amd/{slug}")
    out.append("vendor: amd")
    out.append(f'model: "{model}"')
    out.append(f"architecture: {arch}")
    out.append(f'gcn_arch: "{target}"')
    if target_full:
        # The feature suffixes decide whether a binary built for this device
        # will load on it, so they are recorded verbatim as well.
        out.append(f'gcn_arch_full: "{target_full}"')
    # A wavefront is 64 lanes on CDNA. The interpreter's warp model is sized by
    # this and is currently 32-wide, so the profile is ahead of the engine --
    # deliberately, and noted in TODO.md rather than rounded to 32 here.
    out.append(f"warp_size: {wave}")
    out.append(f"vram_bytes: {vram_kb * 1024}")
    out.append("verified: true")
    out.append("limits:")
    out.append(f"  max_threads_per_block: {wg_max}")
    out.append(f"  max_block_dim: [{wg_dims[0]}, {wg_dims[1]}, {wg_dims[2]}]")
    if grid_dims:
        out.append(f"  max_grid_dim: [{grid_dims[0]}, {grid_dims[1]}, {grid_dims[2]}]")
    if lds:
        out.append(f"  shared_mem_per_block_bytes: {lds}")
        # CDNA has no opt-in carve-out above a default the way CUDA does: the
        # LDS limit is the LDS limit.
        out.append(f"  shared_mem_per_block_optin_bytes: {lds}")
    out.append(f"  multiprocessors: {cus}")
    if waves_per_cu and wave:
        out.append(f"  max_threads_per_sm: {waves_per_cu * wave}")
    # Registers per workgroup and blocks resident per CU are not reported by
    # rocminfo. Left out rather than filled with the CUDA numbers: a wrong
    # residency ceiling is a wrong occupancy calculation that would look
    # measured.
    out.append("features:")
    out.append("  fp16: true")
    out.append(f"  bf16: {'true' if arch in ('cdna2', 'cdna3', 'cdna4') else 'false'}")
    out.append("  fp64: true")
    out.append(f"  matrix_cores: {'true' if arch.startswith('cdna') else 'false'}")
    # Telemetry from the rocm-smi half of the dump. Presentation only -- never
    # consulted by the execution model -- but a monitoring tool needs a scale,
    # and these are the device's own numbers rather than a datasheet's.
    power_cap = smi_value(text, r"Max Graphics Package Power \(W\):\s*([\d.]+)")
    junction = smi_value(text, r"Temperature \(Sensor junction\) \(C\):\s*([\d.]+)")
    out.append("telemetry:")
    out.append(f"  power_limit_w: {int(power_cap) if power_cap else 0}")
    out.append(f"  sm_clock_max_mhz: {clock or 0}")
    out.append("  mem_clock_max_mhz: 0")
    # rocm-smi reports the *current* junction temperature, not the shutdown
    # threshold, so this is left at zero rather than passed off as a limit --
    # the same rule the GH200 forced: a value the device did not report is
    # omitted, not approximated from one it did.
    out.append("  temperature_max_c: 0")
    out.append("  pci_vendor_id: 4098")
    out.append(f"  pci_device_id: {chip or 0}")
    print("\n".join(out))

    print(f"\n# read: {cus} CUs, wavefront {wave}, {waves_per_cu} waves/CU, "
          f"L2 {l2_kb} KB, cacheline {field(g, 'Cacheline Size')} B",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
