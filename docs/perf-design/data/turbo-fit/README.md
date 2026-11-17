# Turbo fit evidence (2026-11-17)

Design and rules: [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) section 15.

## What could not be measured, and why

The card evidence asked for (turbo on the local RTX 3060 with the rtx3060 and the H100 profile, `VGPU_HOST_GPU_VERIFY=all` differing words and bits)
**could not be produced on this machine.** It runs under WSL2, where the mode's driver start cannot work: `cuMemAddressReserve` of the simulator's
window at `0x2000'0000'0000` does not return, the start's 10 s deadline passes and the mode turns itself off
(`the driver did not answer within 10 s ... mode off`; known since the spike, [09-host-gpu-spike.md](../../09-host-gpu-spike.md) section 3, and not changed here).
server1 (RTX 3080 Ti) is closed to workloads and no cloud card was allowed. So no number below comes from a launch that ran on a card by this branch, and none is claimed.
The differing-bit counter (`bits_different`) and `host_gpu_fit:` line are in the summary and unit/stub tested, but have no card reading yet. Open: run
`tools/bench/hostgpu/run.py --forms sim,turbo,verify` (and `--gpu nvidia/h100` on a non-H100 card) on a non-WSL machine with a card.

## What was measured (WSL2, RTX 3060, GPU 0, `VGPU_THREADS=4`, one run each, same programs)

* `before_3060_wsl.txt`: origin/main turbo (rtx3060 profile). `after_3060_wsl.txt`: this branch, turbo with the default `VGPU_HOST_GPU_FIT=1`. Columns: simulator-only wall time, turbo wall time.
  Every turbo run ends "mode off". Before: the driver starts only for the few programs the size rule sends (FDTD-2D, LU, gaussian pay the 10 s wait, +10 s); after: **every program
  pays the 10 s wait, because with FIT the card is started at the first launch**: on a host whose driver start hangs (WSL2), FIT=1 costs 10 s a process (0.01 to 28 s programs become 10 to 38 s). `VGPU_HOST_GPU_FIT=0` is lazy as before.
  Outputs are identical to the simulator's in every row (the mode was off).
* `native_3060.json`: the same binaries run natively on the 3060 (seconds): 2DCONV 0.63, ATAX 0.23, FDTD-2D 5.95, GEMM 0.28, LU 0.81, backprop 0.29, gaussian 0.19, srad_v2 0.21, hotspot 0.18, bfs 0.17
  (process start-up dominates the small ones). The simulator took 0.4 to 28 s for them (before_3060_wsl.txt).
* The H100-profile-on-3060 runs of the brief were not made for the same reason (the mode would be off).

## What was proved with a stand-in driver (no card): `nvidia/tests/e2e/run_host_gpu_stub.sh` steps 45 to 52, run on server3

Clean run: `mut_m5_shared_and_clean_run.txt` (last lines): PASS, exit 0. Unit tests: `test_host_gpu_logic` 126/126.

Mutations (each applied alone, rebuilt, unit test and stub script run; every one fails):

| mutation | file | caught by |
|---|---|---|
| M1 unnamed mnemonic is class never again (`return C::Never` for Unlisted) | include/vgpu/host_gpu.hpp | unit: `turbo_classes_every_kind_of_instruction`, `unlisted_is_turbo_only`, `turbo_whitelist_covers_the_decoder_and_nothing_unnamed`; stub: ELECT kernel refused (`mut_m1_unlisted_never.txt`, 4 FAIL) |
| M2 fit check dropped (`if (fr != None) return fit_refuse` removed) | src/runtime/host_gpu.cpp | stub: every `fl_*` refusal launch reaches `cuLaunchKernel` (`mut_m2_nofit.txt`, 28 FAIL); unit passes (the check is a call, not a function) |
| M3 architecture/SM-count gate restored in turbo | src/runtime/host_gpu.cpp | stub: the H100-profile run, the sm_90 card and the 80-SM card are mode off (`mut_m3_gate.txt`, 41 FAIL) |
| M4 performance rules back under FIT (`fit_send = false`) | src/runtime/host_gpu.cpp | stub: the one-warp launch stays in the simulator (`mut_m4_rules.txt`, 22 FAIL) |
| M5 `fits_card` shared-memory bound `>` to `>=` | include/vgpu/host_gpu_source.hpp | unit: `a_launch_fits_the_card_by_its_limits` (boundary); stub: `fl_shared_ok` |
