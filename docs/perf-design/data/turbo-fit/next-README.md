# Turbo-next evidence (2026-11-17, one A10G rig, sm_86, driver 595, CUDA 13.0 toolkit)

Everything here is in [next-results.md](next-results.md); design and numbers are summarised in [../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md) section 16. The before build is the turbo-fit branch at fae3230b, the after build is this branch. All timings: one rig, `VGPU_THREADS=4`.
The raw outputs (json, md, logs) are kept outside the repository by the person who ran them.

Mutations (each applied alone, unit test and, where noted, the stub script run on a CPU-only machine; each fails): `mut_r1_no_rewrite.txt` (the target line is not rewritten: stub steps for retargeting fail), `mut_r2_blockers_dead.txt`
(retarget blockers never refuse: unit), `mut_p1_no_prestart.txt` (no prestart: stub step 56), `mut_l1_profile_limits.txt` (no profile launch check before the card: stub step 57, the launch with 256 threads of a `__launch_bounds__(128)` kernel reaches the stand-in card).
