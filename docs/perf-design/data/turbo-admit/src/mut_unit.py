#!/usr/bin/env python3
# Mutations of section 20, applied to a copy of the tree on the rig, each built and run against test_host_gpu_logic; the failing test names are the evidence.
import subprocess, sys, shutil, os, re
T=os.path.expanduser('~/par-admit/src-tree')
B=os.path.expanduser('~/par-admit/build')
env=dict(os.environ, PATH='/usr/local/cuda-13.0/bin:'+os.environ['PATH'])
muts=[
 ('m1_every_ptx_variable_unreferenced','include/vgpu/host_gpu_source.hpp','  if (d.name.empty() || d.linked || d.managed) return false;','  if (false) return false;\n  return true;'),
 ('m2_linkage_ignored','include/vgpu/host_gpu_source.hpp','  if (d.name.empty() || d.linked || d.managed) return false;','  if (d.name.empty() || d.managed) return false;'),
 ('m3_managed_ignored','include/vgpu/host_gpu_source.hpp','  if (d.name.empty() || d.linked || d.managed) return false;','  if (d.name.empty() || d.linked) return false;'),
 ('m4_call_statements_not_read','include/vgpu/host_gpu_source.hpp','      if (!decl_stmt.count(si)) ptx_identifiers(stmts[si], 0, used);','      if (!decl_stmt.count(si) && stmts[si].find("call") == std::string::npos) ptx_identifiers(stmts[si], 0, used);'),
 ('m5_initialisers_not_read','include/vgpu/host_gpu_source.hpp','ptx_identifiers(stmts[si], eq + 1, used);   // (an initialiser may name another variable)','(void)eq;'),
 ('m6_name_pieces_not_checked','include/vgpu/host_gpu_source.hpp','  for (const std::string& p : pieces)\n    if (used.count(p)) return false;\n  return true;','  return !used.count(d.name);'),
 ('m7_register_call_join_point_ignored','include/vgpu/host_gpu.hpp','    if (x.entry) return -1;   // a join point between the load and the call','    (void)0;'),
 ('m8_register_call_any_bank','include/vgpu/host_gpu.hpp','x.bank != 4 || ','false || '),
 ('m9_register_call_high_half_write_ignored','include/vgpu/host_gpu.hpp','if (rc_writes(x, t) || rc_writes(x, t + 1)) {','if (rc_writes(x, t)) {'),
 ('m10_guarded_load_accepted','include/vgpu/host_gpu.hpp','if (x.kind != RcInsn::LdcBank || x.guarded || !x.wide','if (x.kind != RcInsn::LdcBank || !x.wide'),
 ('m11_every_bpt_is_a_trap','include/vgpu/host_gpu.hpp','mnemonic == "BPT" && mods.size() == 1 && mods[0] == "TRAP"','mnemonic == "BPT"'),
 ('m12_assert_admitted_when_switched_off','include/vgpu/host_gpu.hpp','(k == ExternKind::Assert && !assert_off)','(k == ExternKind::Assert)'),
 ('m13_assert_admitted_in_exact_mode','include/vgpu/host_gpu.hpp','  return mode == Mode::Turbo && ((k == ExternKind::Printf','  return ((k == ExternKind::Printf'),
]
for name,f,old,new in muts:
    p=os.path.join(T,f)
    src=open(p).read()
    assert old in src, name
    open(p,'w').write(src.replace(old,new,1))
    r=subprocess.run(['cmake','--build',B,'--target','test_host_gpu_logic','-j4'],env=env,capture_output=True,text=True)
    if r.returncode:
        out='BUILD FAILED\n'+r.stdout[-600:]
    else:
        out=subprocess.run([B+'/test_host_gpu_logic'],capture_output=True,text=True).stdout
    fails=[l for l in out.splitlines() if l.startswith('[ FAIL ]') or 'VCHECK' in l]
    print('MUTATION',name); print('\n'.join(fails[:6]) if fails else 'NOT CAUGHT (all tests passed)'); print()
    open(p,'w').write(src)
