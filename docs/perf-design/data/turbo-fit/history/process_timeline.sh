cd $WORK/corp/sp22-corp/rodinia/bin
B=$WORK/build
tl() { # label, env...
  local lab=$1; shift
  env "$@" strace -f -ttt -e trace=execve,openat,exit_group,clone3,execve ./bfs data/graph4096.txt 2>/tmp/tl.txt >/dev/null
  python3 - "$lab" <<'PY'
import re,sys
L=[l for l in open('/tmp/tl.txt')]
def t(l):
    m=re.search(r'(\d{10}\.\d+)',l); return float(m.group(1)) if m else None
ev={}
t0=t(L[0])
first=lambda pat:next((t(l) for l in L if re.search(pat,l) and 'ENOENT' not in l),None)
cu=first('libcuda.so'); nv=first('/dev/nvidiactl'); ex=first('exit_group'); last=t(L[-1])
shim=first('libcudart.so'); res=first('result.txt|graph4096')
f=lambda x:'-' if x is None else '%.0f ms'%((x-t0)*1e3)
print("%-22s libcudart@%s libcuda@%s first /dev/nvidiactl@%s file-open@%s exit_group@%s last_exit@%s"%(sys.argv[1],f(shim),f(cu),f(nv),f(res),f(ex),f(last)))
PY
}
for i in 1 2; do
tl "turbo" LD_LIBRARY_PATH=$B/shim VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_QUIET=1 VGPU_HOST_GPU=turbo
tl "turbo prestart=0" LD_LIBRARY_PATH=$B/shim VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_QUIET=1 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_PRESTART=0
tl "sim" LD_LIBRARY_PATH=$B/shim VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_QUIET=1
tl "native" LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64
done
