# Stage 1: where the AMD engine's work goes (round sp72)

Base build (origin/sp69-hotspot-start 77a8b8c8), one host thread, simulated MI300X. No hardware counters on the host: the shares are of
samples of the instruction pointer, classified by `scripts/classify.py` (its header defines the classes).

## Nanoseconds per wave-instruction

Device CPU time over the exact instruction count of `tools/bench-amd/bench.py` (median of 3 runs, 1-minute load 5 to 6):

| row | wave-instructions | rate | ns per wave-instruction |
|---|---:|---:|---:|
| valu_int | 8,195,840 | 70.1 M/s | 14.3 |
| valu_f32 | 5,637,632 | 63.2 M/s | 15.8 |
| valu_f64 | 8,199,424 | 60.0 M/s | 16.7 |
| divergent | 17,902,446 | 85.4 M/s | 11.7 |
| wave | 7,171,968 | 89.1 M/s | 11.2 |
| lds | 3,100,480 | 31.3 M/s | 31.9 |
| p_i64 | 8,258,560 | 53.2 M/s | 18.8 |
| p_cmp | 13,573,248 | 66.3 M/s | 15.1 |

## The same arithmetic as host code

`scripts/native_ceiling.cpp` (`g++ -O3 -mavx2 -mfma -ffp-contract=off`), nanoseconds a wave (64 lanes) and loop iteration, best of 9, three
passes on cpus 7, 5, 7:

| kernel | pass 1 | pass 2 | pass 3 | wave-instructions an iteration (`VGPU_AMD_BLOCKS_STATS`) |
|---|---:|---:|---:|---:|
| valu_int | 17.27 | 19.27 | 17.11 | 16 |
| valu_f32 | 8.25 | 8.54 | 7.64 | 11 |
| valu_f64 | 7.80 | 7.24 | 7.98 | 8 |

(At `-O2`, and at `-O3` with the integer kernel written as scalar code, g++ does not vectorize the integer loop: 66 and 203 ns.)

## Classes of host work

Bench rows: `HZ=10000 ./prof.sh base/build base <row> 30` (30 runs of the row, 10 kHz). inq: `./inqprof.sh base/build base <test>` (one
run, 1 kHz, `VGPU_THREADS=1`). "outside it" are samples in the program's own code, libc and ROCm's libraries.

### valu_int

```
samples in the library: 3133 (outside it: 6279)
  loop      32.7%
  decode    18.7%
  rows      14.3%
  lanes     34.2%
  mem        0.1%
  interp     0.0%
  other      0.2%
top functions:
   32.7%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
   15.7%  h_int_avx2(tier::Op const&, tier::Regs&)
   12.0%  avx2_impl::avx2_impl_ns::k_bcnt(unsigned int*, unsigned int const*, unsigned int const*, unsigned in
   10.9%  avx2_impl::avx2_impl_ns::k_mad_u64(unsigned int*, unsigned int*, unsigned int const*, unsigned int c
    8.7%  avx2_impl::avx2_impl_ns::k_alignbit(unsigned int*, unsigned int const*, unsigned int const*, unsigne
    3.6%  avx2_impl::avx2_impl_ns::k_xor(unsigned int*, unsigned int const*, unsigned int const*, unsigned int
    3.3%  h_mad_u64_avx2(tier::Op const&, tier::Regs&)
    2.6%  avx2_impl::avx2_impl_ns::k_lshrrev(unsigned int*, unsigned int const*, unsigned int const*, unsigned
    2.5%  avx2_impl::avx2_impl_ns::k_add(unsigned int*, unsigned int const*, unsigned int const*, unsigned int
    1.6%  h_intv_avx2(tier::Op const&, tier::Regs&)
    1.5%  avx2_impl::avx2_impl_ns::k_lshlrev(unsigned int*, unsigned int const*, unsigned int const*, unsigned
    1.3%  avx2_impl::avx2_impl_ns::k_mul_lo(unsigned int*, unsigned int const*, unsigned int const*, unsigned 
    1.0%  avx2_impl::avx2_impl_ns::k_and(unsigned int*, unsigned int const*, unsigned int const*, unsigned int
```

### valu_f32

```
samples in the library: 2497 (outside it: 8186)
  loop      18.0%
  decode    29.3%
  rows      14.7%
  lanes     37.6%
  mem        0.1%
  interp     0.0%
  other      0.3%
top functions:
   48.0%  h_f32_avx2(tier::Op const&, tier::Regs&)
   18.0%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
   13.2%  h_pk_avx2(tier::Op const&, tier::Regs&)
   10.1%  avx2_impl::avx2_impl_ns::k_ffma(unsigned int*, unsigned int const*, unsigned int const*, unsigned in
    4.8%  avx2_impl::avx2_impl_ns::k_fmul(unsigned int*, unsigned int const*, unsigned int const*, unsigned in
    3.7%  avx2_impl::avx2_impl_ns::k_fadd(unsigned int*, unsigned int const*, unsigned int const*, unsigned in
    0.7%  b_scc0(tier::Op const&, tier::Regs&)
    0.5%  int s_add_i32_k<1, 2>(tier::Op const&, tier::Regs&)
    0.2%  int s_cmp_k<0u, 1, 2>(tier::Op const&, tier::Regs&)
    0.2%  table()::{lambda()#1}::operator()() const [clone .isra.0]
    0.1%  set_up_group(Group&, Machine&, Dispat
    0.1%  std::_Rb_tree<std::pair<Enc, unsigned int>, std::pair<std::pair<Enc, unsigned int> const, (anonymous
    0.1%  std::_Rb_tree<std::pair<Enc, unsigned int>, std::pair<std::pair<Enc, unsigned int> const, (anonymous
```

### valu_f64

```
samples in the library: 4328 (outside it: 12465)
  loop      19.6%
  decode    23.1%
  rows      10.1%
  lanes     47.1%
  mem        0.0%
  interp     0.0%
  other      0.1%
top functions:
   41.4%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<2, 1>(unsigned int*, unsigned int*, unsigned int const
   22.0%  h_f64r_avx2(tier::Op const&, tier::Regs&)
   19.5%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
    9.8%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<1, 2>(unsigned int*, unsigned int*, unsigned int const
    4.5%  h_mov64_avx2(tier::Op const&, tier::Regs&)
    1.6%  b_scc0(tier::Op const&, tier::Regs&)
    0.4%  int s_add_i32_k<1, 2>(tier::Op const&, tier::Regs&)
    0.3%  int s_cmp_k<0u, 1, 2>(tier::Op const&, tier::Regs&)
    0.1%  table()::{lambda()#1}::operator()() const [clone .isra.0]
    0.0%  add_comparisons(std::map<std::pair<Enc, unsigned int>, S
    0.0%  std::_Hashtable<std::basic_string_view<char, std::char_traits<char> >, std::pair<std::basic_string_v
    0.0%  int s_mov_k<2, 2>(tier::Op const&, tier::Regs&)
    0.0%  vgpu::(anonymous namespace)::run_in(unsigned char*, unsigned char const*, unsigned long, unsigned in
```

### p_i64

```
samples in the library: 4986 (outside it: 3335)
  loop      22.7%
  decode    16.6%
  rows      18.8%
  lanes     41.7%
  mem        0.0%
  interp     0.0%
  other      0.1%
top functions:
   22.7%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
   14.7%  avx2_impl::avx2_impl_ns::k_mad_u64(unsigned int*, unsigned int*, unsigned int const*, unsigned int c
   13.5%  h_l64_avx2(tier::Op const&, tier::Regs&)
    7.4%  h_mad_u64_avx2(tier::Op const&, tier::Regs&)
    5.7%  avx2_impl::avx2_impl_ns::k_bcnt(unsigned int*, unsigned int const*, unsigned int const*, unsigned in
    5.2%  h_int_avx2(tier::Op const&, tier::Regs&)
    4.8%  avx2_impl::avx2_impl_ns::k_ffbh_u(unsigned int*, unsigned int const*, unsigned int const*, unsigned 
    2.6%  h_cmp64_avx2(tier::Op const&, tier::Regs&)
    2.1%  h_intv_avx2(tier::Op const&, tier::Regs&)
    2.0%  avx2_impl::avx2_impl_ns::k_alignbit(unsigned int*, unsigned int const*, unsigned int const*, unsigne
    1.8%  h_cndmask_avx2(tier::Op const&, tier::Regs&)
    1.8%  h_mov_avx2(tier::Op const&, tier::Regs&)
    1.7%  avx2_impl::avx2_impl_ns::l_lshl_add_g(unsigned long*, unsigned long const*, unsigned long const*, un
```

### divergent

```
samples in the library: 7533 (outside it: 755)
  loop      29.9%
  decode    29.6%
  rows      15.0%
  lanes     25.1%
  mem        0.1%
  interp     0.0%
  other      0.2%
top functions:
   29.7%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
   12.9%  h_mad_u64_avx2(tier::Op const&, tier::Regs&)
   10.1%  h_l64_avx2(tier::Op const&, tier::Regs&)
    7.4%  h_mov_avx2(tier::Op const&, tier::Regs&)
    6.9%  h_mov64_avx2(tier::Op const&, tier::Regs&)
    6.8%  avx2_impl::avx2_impl_ns::k_mad_u64(unsigned int*, unsigned int*, unsigned int const*, unsigned int c
    4.9%  h_cmp64_avx2(tier::Op const&, tier::Regs&)
    3.6%  avx2_impl::avx2_impl_ns::l_lshrrev(unsigned long*, unsigned long const*, unsigned long const*, unsig
    2.9%  h_int_avx2(tier::Op const&, tier::Regs&)
    2.5%  s_or_b64(tier::Op const&, tier::Regs&)
    1.8%  unsigned long avx2_impl::avx2_impl_ns::cmp_i64<2, false>(unsigned long const*, unsigned long const*,
    1.6%  unsigned long avx2_impl::avx2_impl_ns::cmp_i32<2, false>(unsigned int const*, unsigned int const*, u
    1.5%  s_saveexec(tier::Op const&, tier::Regs&)
```

### lds

```
samples in the library: 3827 (outside it: 411)
  loop      34.9%
  decode    31.9%
  rows       1.9%
  lanes      6.9%
  mem       18.2%
  interp     0.0%
  other      6.2%
top functions:
   30.5%  h_ds_avx2(tier::Op const&, tier::Regs&)
   29.3%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
    8.4%  Machine::memory_run(Wave&, Inst const&, bool, unsigned int, unsign
    6.2%  Machine::tier_global(Wave&, Group&, g
    4.8%  Machine::fetch(unsigned long) [clone .lto_priv.0]
    3.7%  vgpu::(anonymous namespace)::run_out(unsigned char const*, unsigned char*, unsigned long, unsigned i
    2.2%  vgpu::(anonymous namespace)::run_in(unsigned char*, unsigned char const*, unsigned long, unsigned in
    1.9%  s_saveexec(tier::Op const&, tier::Regs&)
    1.9%  vgpu::MemoryManager::run_allocation(unsigned long, unsigned long, unsigned int, unsigned long*, unsi
    1.8%  avx2_impl::avx2_impl_ns::k_add(unsigned int*, unsigned int const*, unsigned int const*, unsigned int
    1.5%  s_or_b64(tier::Op const&, tier::Regs&)
    1.3%  h_int_avx2(tier::Op const&, tier::Regs&)
    1.0%  vgpu::MemoryManager::load_run(unsigned long, void*, unsigned long, unsigned int, unsigned char*) con
```

### steepest_descent

```
samples in the library: 30681 (outside it: 2762)
  loop      25.4%
  decode    29.1%
  rows       8.9%
  lanes     21.2%
  mem        6.2%
  interp     3.8%
  other      5.2%
top functions:
   23.5%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
   15.1%  h_f64_avx2(tier::Op const&, tier::Regs&)
    5.7%  h_ds_avx2(tier::Op const&, tier::Regs&)
    3.6%  h_int_sdwa(tier::Op const&, tier::Regs&)
    3.2%  h_cmp64_avx2(tier::Op const&, tier::Regs&)
    2.7%  h_int_avx2(tier::Op const&, tier::Regs&)
    2.3%  h_l64_avx2(tier::Op const&, tier::Regs&)
    2.2%  k_dmax_avx2(unsigned long*, unsigned long const*, unsigned long const*, unsigned long const*, unsign
    1.8%  (anonymous namespace)::drain(std::vector<std::shared_ptr<WorkQueue<hipError_t> >, std::allocator<std
    1.7%  h_cndmask_avx2(tier::Op const&, tier::Regs&)
    1.7%  Machine::fetch(unsigned long) [clone .lto_priv.0]
    1.4%  unsigned long avx2_impl::avx2_impl_ns::cmp_i32<4, false>(unsigned int const*, unsigned int const*, u
    1.2%  std::thread::_State_impl<std::thread::_Invoker<std::tuple<WorkQueue<hipError_t>::submit(std::functio
```

### initial_guess

```
samples in the library: 58096 (outside it: 29574)
  loop      23.6%
  decode    27.8%
  rows       9.1%
  lanes     20.5%
  mem       10.4%
  interp     6.5%
  other      2.3%
top functions:
   21.9%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
   12.9%  h_f64_avx2(tier::Op const&, tier::Regs&)
    5.1%  h_ds_avx2(tier::Op const&, tier::Regs&)
    3.1%  h_cmp64_avx2(tier::Op const&, tier::Regs&)
    2.9%  h_int_sdwa(tier::Op const&, tier::Regs&)
    2.6%  h_int_avx2(tier::Op const&, tier::Regs&)
    2.1%  h_l64_avx2(tier::Op const&, tier::Regs&)
    1.8%  Machine::wave_load4(Wave&, unsigned int, unsigned long const*, Ope
    1.7%  k_dmax_avx2(unsigned long*, unsigned long const*, unsigned long const*, unsigned long const*, unsign
    1.5%  Machine::scratch_access(Wave&, Inst const&, (anonymous 
    1.4%  Machine::fetch(unsigned long) [clone .lto_priv.0]
    1.3%  h_f64r_avx2(tier::Op const&, tier::Regs&)
    1.3%  unsigned long avx2_impl::avx2_impl_ns::cmp_i32<4, false>(unsigned int const*, unsigned int const*, u
```

### poisson

```
samples in the library: 95262 (outside it: 70379)
  loop      22.8%
  decode    22.8%
  rows       5.6%
  lanes     17.3%
  mem       16.4%
  interp     6.0%
  other      9.2%
top functions:
   21.4%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
    7.0%  h_f64r_avx2(tier::Op const&, tier::Regs&)
    4.0%  vgpu::(anonymous namespace)::run_out(unsigned char const*, unsigned char*, unsigned long, unsigned i
    3.4%  set_up_group(Group&, Machine&, Dispat
    3.3%  Machine::memory_run(Wave&, Inst const&, bool, unsigned int, unsign
    2.9%  Machine::tier_global(Wave&, Group&, g
    2.8%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<0, 0>(unsigned int*, unsigned int*, unsigned int const
    2.7%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<1, 0>(unsigned int*, unsigned int*, unsigned int const
    2.6%  h_l64_avx2(tier::Op const&, tier::Regs&)
    2.5%  h_mad_u64_avx2(tier::Op const&, tier::Regs&)
    2.1%  h_ds_avx2(tier::Op const&, tier::Regs&)
    2.1%  h_int_avx2(tier::Op const&, tier::Regs&)
    1.8%  h_mov_avx2(tier::Op const&, tier::Regs&)
```

### ks_hamiltonian

```
samples in the library: 18014 (outside it: 25615)
  loop      16.1%
  decode    18.1%
  rows       6.0%
  lanes     17.3%
  mem       26.2%
  interp     7.0%
  other      9.3%
top functions:
   15.6%  void run_groups<execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, unsigne
    7.1%  h_f64r_avx2(tier::Op const&, tier::Regs&)
    6.6%  Machine::memory_run(Wave&, Inst const&, bool, unsigned int, unsign
    5.8%  vgpu::(anonymous namespace)::run_out(unsigned char const*, unsigned char*, unsigned long, unsigned i
    3.2%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<0, 0>(unsigned int*, unsigned int*, unsigned int const
    3.2%  Machine::wave_load4(Wave&, unsigned int, unsigned long const*, Ope
    3.1%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<2, 0>(unsigned int*, unsigned int*, unsigned int const
    3.0%  Machine::tier_global(Wave&, Group&, g
    3.0%  h_l64_avx2(tier::Op const&, tier::Regs&)
    2.7%  vgpu::MemoryManager::lookup_host_core(unsigned long, unsigned long, vgpu::MemoryManager::HostHit*) c
    2.5%  h_ds_avx2(tier::Op const&, tier::Regs&)
    1.8%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<2, 1>(unsigned int*, unsigned int*, unsigned int const
    1.8%  h_mov64_avx2(tier::Op const&, tier::Regs&)
```

