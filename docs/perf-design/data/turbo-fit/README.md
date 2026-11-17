# Turbo fit evidence (2026-11-17)

Design and rules: [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) section 15.

## Card measurements: [results.md](results.md)

Three AWS rigs (2 x A10G sm_86 with 80 SMs, 1 x L4 sm_89 with 58 SMs; native Linux, driver 595, CUDA 13.0 toolkit), 14 programs (5 PolyBench, 5 Rodinia, 4 HeCBench) built for sm_86, one run each,
`tools/bench/hostgpu/run.py --forms sim,turbo,verify`. The before build is origin/main 37c4cc9d, the after build is this branch; the stand-in tests also passed on rig 1 (unit 127/127, stub script PASS, WSL step included).

Measured (see results.md for every row):
* A10G, profile a10g, turbo FIT=1: geometric mean 1.79x over the simulator (before, same card class: 2.09x for the size-rule turbo; with `VGPU_HOST_GPU_FIT=0` on this branch 2.11x). Long programs gain the most
  (PolyBench FDTD-2D 92.3 s to 4.3 s, LU 28.2 s to 1.0 s, Rodinia gaussian 3.7 s to 0.27 s; native 4.2 s, 0.94 s and 0.30 s).
* **The price of FIT=1 is paid by the small programs**: every program starts the driver, so 2DCONV 0.93 s to 2.0 s (the driver start of that process), bfs, atan2 and srad_v2 0.01 to 0.06 s become 0.24 to 0.5 s.
  The size rule (before, or `FIT=0`) kept those in the simulator. That is why the geometric mean is lower than before although every long program is as fast or faster.
* A10G with the **H100 profile** (sm_90 simulated on the sm_86 card; before: no card path at all, 0 of 7,708 launches, "not a SASS module"): FDTD-2D 198 s to 4.3 s, LU 63 s to 1.1 s, gaussian 8.4 s to 0.27 s, black-scholes 19.5 s to 6.0 s; geometric mean 2.61x
  (before: 1.00x). On the L4 (sm_89, so the lower-minor rule: the card runs the sm_86 cubin) with the H100 profile 2.45x and with the a10g profile (80 simulated SMs on a 58-SM card, which exact mode refuses) 1.69x.
* **Differences from the simulator: none observed.** In all 14 programs on every card and profile, `VGPU_HOST_GPU_VERIFY=all` compared every launch the card ran and found 0 words different (so 0 bits, largest difference 0), and the turbo output equals the simulator's. A single
  summary line read directly (L4, H100 profile, PolyBench GEMM): `host_gpu_turbo: ... words_compared=786432 words_different=0 ... bits_different=0` and `host_gpu_fit: policy=fit card_sm=89 card_sms=58`,
  with the log line `card image sm_86 cubin (not the simulator's image)`. These programs hold no approximate function (black-scholes is the only MUFU kernel), so this does not say what a MUFU-, half- or atomic-heavy program does; unknown.
* **Rodinia srad_v2** (all cards): the kernel faulted on the card (error 700, illegal address: a known out-of-bounds read of the benchmark, which the simulator tolerates and the mirrored window does not), the mode turned itself off and the
  simulator ran the launch; the output is identical. The fallback is the one that existed; the launch was not checked beforehand (the fit check cannot see an address fault).
* Native: A10G 0.17 to 5.3 s, L4 0.23 to 4.5 s (the small ones are process start-up).
* WSL2: the local RTX 3060 under WSL2 cannot serve the mode (the driver's address-window reservation never returns, [09-host-gpu-spike.md](../../09-host-gpu-spike.md)). Before the guard, turbo with FIT=1 waited out the 10 s start deadline in every process there
  (measured: 2DCONV 0.6 s to 10.6 s, mode off). Now turbo with FIT=1 turns itself off at once on a WSL host with a one-line notice (`host_is_wsl`, unit test and stub step 53; `VGPU_HOST_GPU_FIT=0` and exact are untouched).

## What was proved with a stand-in driver (no card): `nvidia/tests/e2e/run_host_gpu_stub.sh` steps 45 to 52, run on server3

Clean run: `mut_m5_shared_and_clean_run.txt` (last lines): PASS, exit 0; unit tests `test_host_gpu_logic` 127/127 (the later WSL step 53 and unit test were run on rig 1: PASS).

Mutations (each applied alone, rebuilt, unit test and stub script run; every one fails):

| mutation | file | caught by |
|---|---|---|
| M1 unnamed mnemonic is class never again (`return C::Never` for Unlisted) | include/vgpu/host_gpu.hpp | unit: `turbo_classes_every_kind_of_instruction`, `unlisted_is_turbo_only`, `turbo_whitelist_covers_the_decoder_and_nothing_unnamed`; stub: ELECT kernel refused (`mut_m1_unlisted_never.txt`, 4 FAIL) |
| M2 fit check dropped (`if (fr != None) return fit_refuse` removed) | src/runtime/host_gpu.cpp | stub: every `fl_*` refusal launch reaches `cuLaunchKernel` (`mut_m2_nofit.txt`, 28 FAIL); unit passes (the check is a call, not a function) |
| M3 architecture/SM-count gate restored in turbo | src/runtime/host_gpu.cpp | stub: the H100-profile run, the sm_90 card and the 80-SM card are mode off (`mut_m3_gate.txt`, 41 FAIL) |
| M4 performance rules back under FIT (`fit_send = false`) | src/runtime/host_gpu.cpp | stub: the one-warp launch stays in the simulator (`mut_m4_rules.txt`, 22 FAIL) |
| M6 `host_is_wsl` never true | include/vgpu/host_gpu_source.hpp | unit: `wsl_is_read_from_the_kernel_version` (`mut_m6_wsl.txt`) |
| M5 `fits_card` shared-memory bound `>` to `>=` | include/vgpu/host_gpu_source.hpp | unit: `a_launch_fits_the_card_by_its_limits` (boundary); stub: `fl_shared_ok` |

## Review fixes (PR #693 review), mutations 2026-11-17 (unit run and, for M11, the stub script, on server3; each alone, each fails)

| mutation | file | caught by |
|---|---|---|
| M7 an unlisted CALL skips the call-target scan (`call_needs_scan` only for class Call) | include/vgpu/host_gpu.hpp | unit `a_call_is_scanned_whatever_its_class` (`mut_m7_call_unscanned.txt`) |
| M8 TMA forms admitted again (names removed from `unmeasured_private_mnemonic`) | include/vgpu/host_gpu.hpp | unit `turbo_classes_every_kind_of_instruction`, `unlisted_is_turbo_only` (`mut_m8_tma_unlisted.txt`) |
| M9 variable layout never mismatches | include/vgpu/host_gpu_image.hpp | unit `the_card_image_must_lay_variables_out_as_the_simulator_s_does` (`mut_m9_layout_ignored.txt`) |
| M10 parameter sizes never mismatch | include/vgpu/host_gpu_image.hpp | unit `the_card_image_must_take_the_simulator_s_arguments` (`mut_m10_params_ignored.txt`) |
| M11 PTX classes not used by the verdict | src/runtime/host_gpu.cpp | stub step 54 (ATOMICS=0 and TURBO_OFF=atomics both reach the card with a PTX float atomic); unit passes (the verdict is a call) (`mut_m11_ptx_classes.txt`) |

The stub steps added: QMMA (sm_89) admitted as unlisted, `TURBO_OFF=unlisted` keeps it out, ELECT (sm_90) class never, the PTX switches (54), variables in an sm_80 card image that is not the simulator's image (55, positive path: layout agrees, no relocation against a variable left). The negative path (a layout that disagrees) is the unit test: nvcc does not produce two builds that disagree.
