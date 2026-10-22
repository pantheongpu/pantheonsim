#!/usr/bin/env bash
# Checks the machine can build PantheonSim, and says what to install when it
# cannot. Run it with `make doctor`.
#
# Required (the exit status is 1 when one is missing):
#   cmake 3.20 or newer, and a C++20 compiler: g++ 12+ or clang++ 17+.
# Reported only (the simulator needs none of them to run):
#   nvcc and hipcc, which compile your own CUDA or HIP programs; the ROCm
#   release; the free memory and the parallel build jobs that fits in it.
#
# Uses bash builtins and the tools it is checking, nothing else, so it also
# runs on a bare machine. PATH decides which tools it finds.
set -u

here=${BASH_SOURCE[0]%/*}
[[ $here == "${BASH_SOURCE[0]}" ]] && here=.
here=$(cd "$here" && pwd)
missing=0

ok()    { printf '  ok        %s\n' "$*"; }
note()  { printf '  note      %s\n' "$*"; }
gone()  { printf '  MISSING   %s\n' "$*"; missing=1; }
hint()  { printf '            %s\n' "$*"; }

os_id=""; os_ver=""
if [[ -r /etc/os-release ]]; then
  while IFS='=' read -r k v; do
    v=${v%\"}; v=${v#\"}
    [[ $k == ID ]] && os_id=$v
    [[ $k == VERSION_ID ]] && os_ver=$v
  done < /etc/os-release
fi

have() { command -v "$1" >/dev/null 2>&1; }

echo "PantheonSim build check"

# ---- cmake -----------------------------------------------------------------
if have cmake; then
  out=$(cmake --version 2>/dev/null | { read -r l; echo "$l"; })
  if [[ $out =~ ([0-9]+)\.([0-9]+)(\.[0-9]+)? ]]; then
    major=${BASH_REMATCH[1]}; minor=${BASH_REMATCH[2]}
    if (( major > 3 || (major == 3 && minor >= 20) )); then
      ok "cmake ${major}.${minor} (needs 3.20 or newer)"
    else
      gone "cmake ${major}.${minor} is too old (needs 3.20 or newer)"
      hint "sudo apt-get install cmake        (Ubuntu 22.04 has 3.22, 24.04 has 3.28)"
      hint "older releases: pip install cmake, or the installer from cmake.org"
    fi
  else
    gone "cmake is installed but its version could not be read"
  fi
else
  gone "cmake not found (needs 3.20 or newer)"
  hint "sudo apt-get install cmake"
fi

# ---- C++20 compiler --------------------------------------------------------
good_cxx=""
old_cxx=""
for cxx in "${CXX:-}" g++ g++-14 g++-13 g++-12 clang++ clang++-19 clang++-18 clang++-17; do
  [[ -n $cxx ]] && have "$cxx" || continue
  v=""
  if [[ $cxx == *clang* ]]; then
    out=$("$cxx" --version 2>/dev/null | { read -r l; echo "$l"; })
    [[ $out =~ version[[:space:]]+([0-9]+) ]] && v=${BASH_REMATCH[1]}
    need=17; kind="clang++"
  else
    out=$("$cxx" -dumpversion 2>/dev/null)
    [[ $out =~ ^([0-9]+) ]] && v=${BASH_REMATCH[1]}
    need=12; kind="g++"
  fi
  [[ -n $v ]] || continue
  if (( v >= need )); then
    good_cxx="$cxx ($kind $v)"; good_name=$cxx
    break
  fi
  [[ -n $old_cxx ]] || old_cxx="$cxx ($kind $v, needs $need or newer)"
done
if [[ -n $good_cxx ]]; then
  ok "C++20 compiler: $good_cxx"
  # scripts/build.sh uses the compiler cmake picks, which is the default c++.
  if [[ -z ${CXX:-} && $good_name != g++ && $good_name != clang++ ]]; then
    note "build with it: export CXX=$good_name   (cmake would pick the default c++ otherwise)"
  fi
else
  if [[ -n $old_cxx ]]; then
    gone "no C++20 compiler new enough; found $old_cxx"
  else
    gone "no C++ compiler found (needs g++ 12+ or clang++ 17+)"
  fi
  if [[ $os_id == ubuntu && $os_ver == 22.04 ]]; then
    hint "Ubuntu 22.04 ships g++ 11. Install g++ 12 from the main archive:"
    hint "  sudo apt-get install g++-12"
    hint "then build with: export CXX=g++-12"
  elif [[ $os_id == ubuntu ]]; then
    hint "sudo apt-get install g++-12        (then: export CXX=g++-12 if g++ is older)"
  else
    hint "install g++ 12 or newer (apt: g++-12, dnf: gcc-c++) or clang 17 or newer"
  fi
fi

# ---- optional: the compilers for your own programs ---------------------------
if have nvcc; then
  out=$(nvcc --version 2>/dev/null)
  if [[ $out =~ release[[:space:]]+([0-9]+\.[0-9]+) ]]; then
    ok "nvcc ${BASH_REMATCH[1]} (compiles CUDA programs; build them with -cudart shared)"
  else
    ok "nvcc found (version not read)"
  fi
else
  note "nvcc not found: only needed to compile your own CUDA programs"
  hint "sudo apt-get install nvidia-cuda-toolkit   or NVIDIA's CUDA repository"
fi
if have hipcc; then
  out=$(hipcc --version 2>/dev/null | { read -r l; echo "$l"; })
  if [[ $out =~ ([0-9]+\.[0-9]+(\.[0-9]+)?) ]]; then
    ok "hipcc ${BASH_REMATCH[1]} (compiles HIP programs for AMD GPUs)"
  else
    ok "hipcc found (version not read)"
  fi
else
  note "hipcc not found: only needed to compile your own HIP programs"
  hint "AMD's ROCm repository (rocm.docs.amd.com); PantheonSim supports ROCm 6.4 and 7.x"
fi
rocm=""
for f in "${ROCM_PATH:-/opt/rocm}/.info/version" /opt/rocm/.info/version; do
  if [[ -r $f ]]; then read -r rocm < "$f"; break; fi
done
[[ -n $rocm ]] && ok "ROCm release ${rocm} installed"

# ---- memory ------------------------------------------------------------------
avail_kib=""
if [[ -r /proc/meminfo ]]; then
  while read -r k v _; do
    [[ $k == MemAvailable: ]] && { avail_kib=$v; break; }
  done < /proc/meminfo
fi
if [[ -n $avail_kib ]]; then
  gib=$(( avail_kib / 1048576 )); tenth=$(( (avail_kib % 1048576) * 10 / 1048576 ))
  jobs=""
  if [[ -x $here/../ci/jobs.sh ]]; then jobs=$("$here/../ci/jobs.sh" -p release 2>/dev/null); fi
  line="memory: ${gib}.${tenth} GiB available"
  [[ -n $jobs ]] && line+="; the build will use $jobs parallel job(s) (ci/jobs.sh)"
  if (( avail_kib < 4 * 1048576 )); then
    note "$line"
    hint "a Release build links with LTO and wants about 4 GiB free; it may fail or be slow"
    hint "close other programs, or build in a Debug tree: VGPU_BUILD_TYPE=Debug make build"
  else
    ok "$line"
  fi
else
  note "memory: could not be read (/proc/meminfo)"
fi

echo
if (( missing )); then
  echo "Not ready: install what is marked MISSING, then run 'make doctor' again."
  exit 1
fi
echo "Ready: 'make build' builds the simulator, then 'make demo' runs a first program."
exit 0
