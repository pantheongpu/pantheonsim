# What may differ between NVIDIA profiles, for nvidia/tests/e2e/run_xarch.sh (sourced by it). xarch_groups[<program>] lists the profiles that
# must give one answer, groups separated by "|": profiles in different groups may differ, and every difference has a reason below, with the
# evidence for it. <program>@ptx and <program>@native hold in that mode only. A program not here must agree on every profile that runs it.
# Profiles: t4 (sm_75), a100 (sm_80), a10g (sm_86), l4 (sm_89), h100 and h200 (sm_90), b200 (sm_100), rtx5090 (sm_120).

# --- real hardware: the tensor core's sum (docs/numerics/per-profile.md): Turing and Ampere and Ada "ampere" (chunks of 8, tf32 4), Hopper and
# Blackwell "hopper" (chunks of 16, tf32 8). t4 is measured, a10g and l4 measured, h100 inferred from vectors others recorded (arXiv 2512.07004).
tc="t4 a100 a10g l4 | h100 h200 b200 rtx5090"
xarch_groups[float_half_forms]="$tc"      # mma.sync / wmma results (nvidia/tests/data/float_half_forms.h100.txt, .rtx3060.txt)
xarch_groups[simd_lane_forms]="$tc"       # nvidia/tests/data/simd_lane_forms.{t4,rtx3060,h100}.txt
xarch_groups[round9_probe@ptx]="$tc"
xarch_groups[round9_probe@native]="a100 a10g l4 | h100 h200 | b200 rtx5090"   # the probe builds other f64 and NaN forms for sm_100 and sm_120 (__CUDA_ARCH__); the PTX engine agrees across them

# --- real hardware: what the architecture has.
xarch_groups[mma_accumulate_probe]="t4 a100 a10g l4 h100 h200 b200"   # block-scaled mma.sync exists on compute capability 12 only (it SKIPs on b200)
xarch_groups[mma_forms@native]="a100 a10g l4 h100 h200 b200"          # the same: sm_120's block-scaled forms are built there only
xarch_groups[vector_atomics@native]="t4 a100 a10g l4 | h100 h200 b200 rtx5090"   # red/atom .v2/.v4 need sm_90 (the program says SKIP below it)
xarch_groups[smem_size_regs@native]="t4 a100 a10g l4 | h100 h200 b200 rtx5090"   # %aggr_smem_size is from sm_90 (PTX ISA 7.8)
xarch_groups[launch_refusals]="t4 a100 a10g l4 | h100 h200 b200 rtx5090"          # clusters are from sm_90: cudaErrorInvalidClusterSize below
xarch_groups[device_limits]="t4 a100 a10g l4 | h100 h200 b200 rtx5090"           # cudaLimitDevRuntimeSyncDepth is unsupported from 9.0 (CUDA docs)
xarch_groups[shared_max]="t4 | a100 | a10g l4 rtx5090 | h100 h200 b200"          # the opt-in shared memory per block: 64, 163, 99, 227 KiB (CUDA programming guide, compute capability table)
xarch_groups[reserved_smem_attr]="t4 | a100 | a10g l4 rtx5090 | h100 h200 b200"  # the same carveouts
# A program that builds PTX for compute_80 loads on a Turing card only if it is for sm_75 or below: the T4 refuses the module (a Tesla T4 card
# run of launch_cache and driver_entry_late, 2026-11-23, answers as the t4 profile does).
xarch_groups[launch_cache]="a100 a10g l4 h100 h200 b200 rtx5090"
xarch_groups[driver_entry_late]="a100 a10g l4 h100 h200 b200 rtx5090"
xarch_groups[memo_api_matrix@native]="a100 a10g l4 h100 h200 b200 rtx5090"          # a cooperative launch's size follows the SM count; the T4's differs

# --- programs that ask the device what it is: SM count, memory, resident blocks. Profiles of one generation with the same SM count agree.
all_apart="t4 | a100 | a10g | l4 | h100 | h200 | b200 | rtx5090"
for p in limits_probe async_launch_events cdp_device_api device_graph_launch leak_soak lockstep_blocks occupancy_rules; do xarch_groups[$p]="$all_apart"; done
xarch_groups[cluster_occupancy]="a10g l4 | h100 h200"      # the cluster counts follow the GPC layout of each profile (nvidia/docs/clusters.md); #750
xarch_groups[smid]="h100 h200"
xarch_groups[libraries]="h100 h200"                         # the names of the CUB kernels carry the architecture
xarch_groups[sass_archs@ptx]="t4 a100 a10g l4 | h100 h200 | b200 rtx5090"      # the program tests what each generation has (clusters, CLC from sm_100)
xarch_groups[sass_archs@native]="a100 a10g l4 | h100 h200 | b200 rtx5090"      # (mbarrier from sm_80: the T4 build runs other kernels)

# --- programs that read the device's compute capability at run time and gate their code on __CUDA_ARCH__ at build time: the PTX mode runs a build for
# a lower architecture on a higher device, which tries forms the build lacks. In native mode each profile runs its own build, so they differ only by form.
xarch_groups[minmax_forms@ptx]="a10g l4 h100 h200 | b200 rtx5090"
xarch_groups[minmax_forms@native]="a10g l4 h100 h200 | b200 rtx5090"   # a100 lacks the xorsign forms (sm_86), b200 and rtx5090 the 3-input forms (sm_100)

# --- test programs whose output depends on host threads or on the order of host callbacks: their memory digests are not comparable.
xarch_nodigest=" sched_pool "

# --- known defects, each tracked: the profile listed last differs and should not. Delete the entry when its issue is fixed.
xarch_groups[bulk_copy@native]="h100 h200 b200"             # sm_120's cp.async.bulk multicast is a call to __cuda_syscall_cp_async_bulk_multicast the SASS engine lacks; #751
xarch_groups[cg_multi_warp_tiles@native]="a100 a10g l4 | h100 h200 | b200 rtx5090"   # %reserved_smem_offset_* read 0 in SASS from sm_90 on; needs an H100; #752
xarch_groups[texture_grad@native]="t4 a100 a10g l4 | h100 h200 | b200 rtx5090"       # sm_90+ SASS: 37 of 131 tex.grad.3d cases differ from the PTX engine and the RTX 3060; needs an H100; #753
xarch_groups[graph_capture_driver@native]="t4 a100 a10g l4 h100 h200 b200"           # sm_120's code makes other last bits (ptxas contracts differently); not a defect of the simulator; #754
