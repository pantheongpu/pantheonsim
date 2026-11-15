# What is left: samples of four inq tests (round sp74)

`inqprof.sh <build> <tag> <tests>` (round sp72's): one host thread, round sp69's sampler at 1 kHz (a sample is a millisecond of CPU), the
samples inside the simulator's library put in classes by `classify.py` (round sp72's; its header says what each class is: *loop* is the
dispatch of one instruction after another and its counters, *decode* the scalar host instructions of the handlers -- operand kinds, the
EXEC tests, the calls of the lane kernels -- *rows* the loads and stores of register rows, *lanes* the guest instruction's own
arithmetic, *mem* the memory instructions and the memory manager, *interp* the interpreter's own code, *other* launch, set-up and the
API). *outside*: the test's own host code, the ROCm runtime, libc. The split between decode, rows and lanes is approximate.

## Milliseconds of CPU by class

| test | build | whole process | outside | library | loop | decode | rows | lanes | mem | interp | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| steepest_descent | base | 25476 | 1649 | 23827 | 6100 | 7053 | 2192 | 5099 | 1358 | 858 | 1191 |
| steepest_descent | final | 20332 | 1456 | 18876 | 5984 | 4832 | 1718 | 2907 | 1491 | 868 | 1076 |
| steepest_descent | final, blocks on | 19204 | 1448 | 17756 | 4688 | 5203 | 1509 | 2788 | 1492 | 888 | 1207 |
| initial_guess | base | 65082 | 21077 | 44005 | 10429 | 12145 | 4136 | 9197 | 4224 | 2948 | 924 |
| initial_guess | final | 58289 | 21372 | 36917 | 10521 | 9118 | 3175 | 5870 | 4356 | 2880 | 997 |
| initial_guess | final, blocks on | 54888 | 20290 | 34598 | 9065 | 8788 | 3114 | 5639 | 4290 | 2906 | 830 |
| poisson | base | 116720 | 48260 | 68460 | 15335 | 15472 | 4039 | 12254 | 11433 | 3834 | 6093 |
| poisson | final | 111575 | 47829 | 63746 | 15108 | 15554 | 3187 | 8669 | 11411 | 4016 | 5865 |
| poisson | final, blocks on | 107945 | 47827 | 60118 | 10521 | 16713 | 2765 | 8116 | 11843 | 3968 | 6192 |
| ks_hamiltonian | base | 26827 | 15876 | 10951 | 1599 | 1916 | 635 | 1960 | 3296 | 712 | 821 |
| ks_hamiltonian | final | 26032 | 15816 | 10216 | 1624 | 2002 | 480 | 1583 | 2748 | 736 | 1042 |
| ks_hamiltonian | final, blocks on | 25652 | 15758 | 9894 | 1167 | 2117 | 475 | 1494 | 2731 | 792 | 1108 |

## The same as shares of the whole process, final library, knob off

| test | outside | loop | decode | rows | lanes | mem | interp | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steepest_descent | 7 % | 29 % | 24 % | 8 % | 14 % | 7 % | 4 % | 5 % |
| initial_guess | 37 % | 18 % | 16 % | 5 % | 10 % | 7 % | 5 % | 2 % |
| poisson | 43 % | 14 % | 14 % | 3 % | 8 % | 10 % | 4 % | 5 % |
| ks_hamiltonian | 61 % | 6 % | 8 % | 2 % | 6 % | 11 % | 3 % | 4 % |

## steepest_descent, final: the functions with the most samples (share of the library's)

```
   29.7%  void run_groups<false, execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, 
    7.0%  h_ds_avx2(Op const&, Regs&)
    6.1%  int h_dminmax_rows<true>(Op const&, Regs&)
    3.9%  h_cmp64_rows(Op const&, Regs&)
    3.4%  h_int_avx2(Op const&, Regs&)
    2.4%  h_l64_rows(Op const&, Regs&)
    2.3%  h_cndmask_avx2(Op const&, Regs&)
    2.0%  drain(std::vector<std::shared_ptr<WorkQueue<hipError_t> >, std::allocator<std::shared_ptr<WorkQueue<hipEr
    1.8%  Machine::fetch(unsigned long) [clone .lto_priv.0]
    1.6%  unsigned long avx2_impl::avx2_impl_ns::cmp_i32<4, false>(unsigned int const*, unsigned int const*, unsigned 
    1.5%  vgpu::MemoryManager::lookup_host_core(unsigned long, unsigned long, vgpu::MemoryManager::HostHit*) const
    1.5%  std::thread::_State_impl<std::thread::_Invoker<std::tuple<WorkQueue<hipError_t>::submit(std::function<hipErr
```

## steepest_descent, final, blocks on: the functions with the most samples (share of the library's)

```
   17.7%  run_wave_blocks(Machine&, Wave&, (ano
    7.5%  h_ds_avx2(Op const&, Regs&)
    6.0%  int h_dminmax_rows<true>(Op const&, Regs&)
    5.4%  run_round_live(Machine&, Group&, unsigned long) 
    4.2%  h_cmp64_rows(Op const&, Regs&)
    3.6%  h_int_avx2(Op const&, Regs&)
```

## initial_guess, final: the functions with the most samples (share of the library's)

```
   26.4%  void run_groups<false, execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, 
    6.2%  h_ds_avx2(Op const&, Regs&)
    5.2%  int h_dminmax_rows<true>(Op const&, Regs&)
    3.7%  h_cmp64_rows(Op const&, Regs&)
    3.3%  h_int_avx2(Op const&, Regs&)
    2.1%  Machine::wave_load4(Wave&, unsigned int, unsigned long const*, gcn::Ope
    2.0%  h_l64_rows(Op const&, Regs&)
    1.9%  Machine::scratch_access(Wave&, gcn::Inst const&, (anonymous 
    1.6%  unsigned long avx2_impl::avx2_impl_ns::cmp_i32<4, false>(unsigned int const*, unsigned int const*, unsigned 
    1.6%  Machine::fetch(unsigned long) [clone .lto_priv.0]
    1.4%  h_f64r_avx2(Op const&, Regs&)
    1.4%  avx2_impl::avx2_impl_ns::k_mad_u64(unsigned int*, unsigned int*, unsigned int const*, unsigned int const*, u
```

## initial_guess, final, blocks on: the functions with the most samples (share of the library's)

```
   19.9%  run_round_live(Machine&, Group&, unsigned long) 
    6.6%  h_ds_avx2(Op const&, Regs&)
    5.1%  int h_dminmax_rows<true>(Op const&, Regs&)
    3.8%  run_wave_blocks(Machine&, Wave&, (ano
    3.6%  h_cmp64_rows(Op const&, Regs&)
    3.5%  h_int_avx2(Op const&, Regs&)
```

## poisson, final: the functions with the most samples (share of the library's)

```
   22.3%  void run_groups<false, execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, 
    7.3%  h_f64r_avx2(Op const&, Regs&)
    4.2%  vgpu::run_out(unsigned char const*, unsigned char*, unsigned long, unsigned int)
    3.3%  Machine::memory_run(Wave&, gcn::Inst const&, bool, unsigned int, unsign
    3.2%  set_up_group(Group&, Machine&, Dispat
    2.8%  Machine::tier_global(Wave&, Group&, g
    2.7%  h_mad_u64_avx2(Op const&, Regs&)
    2.4%  h_l64_rows(Op const&, Regs&)
    2.3%  h_ds_avx2(Op const&, Regs&)
    2.3%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<0, 0>(unsigned int*, unsigned int*, unsigned int const*, unsig
    2.2%  h_int_avx2(Op const&, Regs&)
    1.9%  h_mov_avx2(Op const&, Regs&)
```

## poisson, final, blocks on: the functions with the most samples (share of the library's)

```
   10.4%  run_wave_blocks(Machine&, Wave&, (ano
    8.3%  h_f64r_avx2(Op const&, Regs&)
    5.1%  run_round_live(Machine&, Group&, unsigned long) 
    5.0%  vgpu::run_out(unsigned char const*, unsigned char*, unsigned long, unsigned int)
    3.7%  Machine::memory_run(Wave&, gcn::Inst const&, bool, unsigned int, unsign
    3.5%  set_up_group(Group&, Machine&, Dispat
```

## ks_hamiltonian, final: the functions with the most samples (share of the library's)

```
   15.2%  void run_groups<false, execute(Dispatch const&, vgpu::MemoryManager&)::{lambda(unsigned long, 
    7.8%  Machine::memory_run(Wave&, gcn::Inst const&, bool, unsigned int, unsign
    7.5%  h_f64r_avx2(Op const&, Regs&)
    6.4%  vgpu::run_out(unsigned char const*, unsigned char*, unsigned long, unsigned int)
    3.4%  Machine::wave_load4(Wave&, unsigned int, unsigned long const*, gcn::Ope
    3.4%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<0, 0>(unsigned int*, unsigned int*, unsigned int const*, unsig
    2.8%  Machine::tier_global(Wave&, Group&, g
    2.7%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<2, 0>(unsigned int*, unsigned int*, unsigned int const*, unsig
    2.6%  h_ds_avx2(Op const&, Regs&)
    2.6%  vgpu::MemoryManager::lookup_host_core(unsigned long, unsigned long, vgpu::MemoryManager::HostHit*) const
    2.4%  h_l64_rows(Op const&, Regs&)
    2.0%  vgpu::MemoryManager::host_load_batch(unsigned long const*, unsigned long, unsigned int, unsigned long*) cons
```

## ks_hamiltonian, final, blocks on: the functions with the most samples (share of the library's)

```
    8.8%  run_wave_blocks(Machine&, Wave&, (ano
    8.6%  h_f64r_avx2(Op const&, Regs&)
    7.8%  Machine::memory_run(Wave&, gcn::Inst const&, bool, unsigned int, unsign
    7.4%  vgpu::run_out(unsigned char const*, unsigned char*, unsigned long, unsigned int)
    3.7%  Machine::wave_load4(Wave&, unsigned int, unsigned long const*, gcn::Ope
    3.5%  unsigned long avx2_impl::avx2_impl_ns::k_f64r<0, 0>(unsigned int*, unsigned int*, unsigned int const*, unsig
```
