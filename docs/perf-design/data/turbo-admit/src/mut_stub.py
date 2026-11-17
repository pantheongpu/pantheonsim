#!/usr/bin/env python3
# Mutations of the runtime (src/runtime/host_gpu.cpp), each built and run against step 59 of the stub e2e (the steps for assert, trap and PTX variables, run alone)
import subprocess, os
T=os.path.expanduser('~/par-admit/src-tree')
B=os.path.expanduser('~/par-admit/build')
env=dict(os.environ, PATH='/usr/local/cuda-13.0/bin:'+os.environ['PATH'], VGPU_BUILD_DIR=B)
f=T+'/src/runtime/host_gpu.cpp'
muts=[
 ('s1_assert_and_trap_never_admitted','const bool trap_ok = turbo && !sw.assert_off;','const bool trap_ok = false;'),
 ('s2_a_card_assert_is_an_ordinary_failure','if (dec.trap && (r == kCudaErrorAssert || r == kCudaErrorLaunchFailure)) {','if (false) {'),
 ('s3_any_card_error_is_taken_for_the_programs_own','if (dec.trap && (r == kCudaErrorAssert || r == kCudaErrorLaunchFailure)) {','if (dec.trap) {'),
 ('s4_ptx_variables_always_refuse','bool ptx_vars_unreferenced = true;','bool ptx_vars_unreferenced = false;'),
]
for name,old,new in muts:
    p=f
    if name.startswith('s4'): p=T+'/include/vgpu/host_gpu.hpp'
    src=open(p).read(); assert old in src, name
    open(p,'w').write(src.replace(old,new,1))
    r=subprocess.run(['cmake','--build',B,'--target','vgpu','vgpucudart','-j4'],env=env,capture_output=True,text=True)
    if r.returncode: out='BUILD FAILED '+r.stdout[-400:]
    else:
        out=subprocess.run([T+'/nvidia/tests/e2e/step59.sh'],env=env,capture_output=True,text=True).stdout
    lines=[l[:330] for l in out.splitlines() if l.startswith('FAIL') or l.startswith('PASS')]
    print('MUTATION',name); print('\n'.join(lines[:5]) if lines else out[-300:]); print()
    open(p,'w').write(src)
r=subprocess.run(['cmake','--build',B,'--target','vgpu','vgpucudart','-j4'],env=env,capture_output=True,text=True)
print('restored build:', r.returncode)
out=subprocess.run([T+'/nvidia/tests/e2e/step59.sh'],env=env,capture_output=True,text=True).stdout
print('UNMUTATED', [l for l in out.splitlines() if l.startswith('FAIL') or l.startswith('PASS')])
