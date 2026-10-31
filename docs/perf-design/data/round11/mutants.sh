#!/bin/bash
# runs ON THE RIG (host): mutation checks. Each mutant: edit ~/jit/src with sed, build into build-mut (cpus 5,6,7,13,14,15), run the unit test and
# the e2e on sass_tier.sm_86 + sass_tier_steps.sm_86, record whether anything failed, then restore the file.
cd ~/jit
CPUS=5,6,7,13,14,15
sudo rm -f ~/results/jit-mut.log
log() { echo "$@" | sudo tee -a ~/results/jit-mut.log >/dev/null; }
[ -d build-mut ] || sudo cp -a build build-mut
declare -A FILE SED
FILE[m1]=src/sass/jit.cpp;      SED[m1]='s|\*c->thread_instructions += lanes;|*c->thread_instructions += glanes;|'
FILE[m2]=src/sass/jit.hpp;      SED[m2]='s|n + k <= kRunLimit|n + k <= kRunLimit + 1|'
FILE[m3]=src/sass/jit.cpp;      SED[m3]='s#if (op.form == tier::Form::Stg || op.ins->op == Op::STG || op.ins->op == Op::ST) return false;#(void)0;#'
FILE[m4]=src/sass/jit.hpp;      SED[m4]='s#steps + k <= max_steps \&\& ##'
FILE[m9]=src/sass/jit.hpp;      SED[m9]='s# \&\& maxreg <= nregs##'
FILE[m5]=src/sass/tier_ops.inc; SED[m5]='s|^  h.last = \&op;$||'
FILE[m6]=src/sass/jit.cpp;      SED[m6]='s#if (ex | (c->flags \& 1u)) c->escape#if (ex) c->escape#'
FILE[m7]=src/sass/jit.cpp;      SED[m7]='s#if (o.guard_not) s += "    g = ~g;\\n";#(void)0;#'
FILE[m8]=src/sass/tier_ops.inc; SED[m8]='s|if (faulting) cur = faulting;||'
for m in ${MUTANTS:-m1 m2 m3 m4 m5 m6 m7 m8}; do
  f=${FILE[$m]}
  cp $f /tmp/$m.orig
  sed -i "${SED[$m]}" $f
  if diff -q $f /tmp/$m.orig >/dev/null; then log "$m: SED DID NOT CHANGE THE FILE"; continue; fi
  ~/drx.sh jit $CPUS "cmake --build build-mut -j6 2>&1 | grep -E ' error |FAILED' | head -5" 2>&1 | sudo tee -a ~/results/jit-mut.log >/dev/null
  log "$m: built; $(diff $f /tmp/$m.orig | head -4 | tr '\n' ' ')"
  u=$(~/drx.sh jit $CPUS "build-mut/test_jit 2>&1 | tail -1")
  log "$m: test_jit: $u"
  ~/drx.sh jit $CPUS "VGPU_JIT_PROGRAMS='sass_tier.sm_86 sass_tier_steps.sm_86' VGPU_TIER_WORK=/w/build-work VGPU_BUILD_DIR=/w/build-mut nvidia/tests/e2e/run_sass_jit.sh 2>&1 | grep -E '^(FAIL|PASS|ok)' | cut -c1-400" 2>&1 | sudo tee -a ~/results/jit-mut.log >/dev/null
  cp /tmp/$m.orig $f
done
log "mutants done"
