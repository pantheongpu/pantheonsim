#!/usr/bin/env bash
# Installs the tools `make build` needs, when they are missing, and nothing else.
# Run it with `make deps` (which `make` does first).
#
# Needed (the same list as scripts/doctor.sh): cmake 3.20 or newer, and a C++20
# compiler: g++ 12+ or clang++ 17+. It never installs CUDA or ROCm (docs/toolchains.md).
#
# What it will and will not do:
#   * Nothing missing: says so and exits 0.
#   * Something missing on Debian or Ubuntu: lists the packages and asks
#     (default no). YES=1 answers yes. With no terminal on stdin (CI, a script) it
#     installs nothing and exits 1 with the exact command to run, so it never
#     waits for an answer and never changes a system unasked.
#   * Not root: uses sudo when there is one. Without it, prints the command.
#   * Ubuntu 20.04 and older have no g++ 12 in the main archive (22.04 does). Adding a PPA is
#     a bigger step, so it is only done with YES=1 PPA=1; otherwise the three
#     commands are printed.
#   * dnf, pacman, zypper: the exact command is printed, never run.
#
#   scripts/deps.sh           check, then install what is missing (asks first)
#   scripts/deps.sh --check   only report; exit 1 when something is missing
#   scripts/deps.sh --cxx     print the compiler the build should use when the default is too old
#                             (nothing when the default is fine)
# Environment: YES=1, PPA=1, CXX (a compiler to prefer), DEPS_OS_RELEASE (test hook).
set -u

mode=install
case ${1:-} in
  "") ;;
  --check) mode=check ;;
  --cxx) mode=cxx ;;
  -h|--help) sed -n '2,/^set -u/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
  *) echo "deps: unknown option '$1'" >&2; exit 2 ;;
esac

have() { command -v "$1" >/dev/null 2>&1; }
say() { [[ $mode == cxx ]] || printf '%s\n' "$*"; }

os_id=""; os_like=""; os_ver=""
if [[ -r ${DEPS_OS_RELEASE:-/etc/os-release} ]]; then
  while IFS='=' read -r k v; do
    v=${v%\"}; v=${v#\"}
    case $k in ID) os_id=$v ;; ID_LIKE) os_like=$v ;; VERSION_ID) os_ver=$v ;; esac
  done < "${DEPS_OS_RELEASE:-/etc/os-release}"
fi
os_major=${os_ver%%.*}
[[ $os_major =~ ^[0-9]+$ ]] || os_major=0

# ---- what is there ----------------------------------------------------------
cmake_ok() {
  have cmake || return 1
  local out
  out=$(cmake --version 2>/dev/null | { read -r l; echo "$l"; })
  [[ $out =~ ([0-9]+)\.([0-9]+) ]] || return 1
  (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] >= 20) ))
}

# cxx_good <compiler>: it exists and speaks C++20 well enough (g++ 12, clang++ 17).
cxx_good() {
  have "$1" || return 1
  local out v
  out=$("$1" --version 2>/dev/null | { read -r l; echo "$l"; })
  if [[ $out == *clang* ]]; then
    [[ $out =~ version[[:space:]]+([0-9]+) ]] || return 1
    (( BASH_REMATCH[1] >= 17 ))
  else
    v=$("$1" -dumpversion 2>/dev/null)
    [[ $v =~ ^([0-9]+) ]] || return 1
    (( BASH_REMATCH[1] >= 12 ))
  fi
}

# The compiler cmake picks without being told: $CXX, else c++.
default_cxx=${CXX:-c++}
good_cxx=""
if cxx_good "$default_cxx"; then
  good_cxx=$default_cxx; cxx_is_default=1
else
  cxx_is_default=0
  for c in g++ g++-14 g++-13 g++-12 clang++ clang++-19 clang++-18 clang++-17; do
    cxx_good "$c" && { good_cxx=$c; break; }
  done
fi

if [[ $mode == cxx ]]; then
  (( cxx_is_default )) || [[ -z $good_cxx ]] || echo "$good_cxx"
  exit 0
fi

need_cmake=0; need_cxx=0
cmake_ok || need_cmake=1
[[ -n $good_cxx ]] || need_cxx=1
if (( ! need_cmake && ! need_cxx )); then
  say "Build dependencies are installed (cmake and a C++20 compiler${good_cxx:+: $good_cxx})."
  exit 0
fi

# ---- what to install -----------------------------------------------------------
say "Missing for 'make build':"
(( need_cmake )) && say "  cmake 3.20 or newer"
(( need_cxx ))   && say "  a C++20 compiler (g++ 12 or newer, or clang++ 17 or newer)"

family=""
for w in $os_id $os_like; do
  case $w in
    debian|ubuntu) family=apt; break ;;
    fedora|rhel|centos|rocky|almalinux) family=dnf; break ;;
    arch) family=pacman; break ;;
    suse|opensuse*|sles) family=zypper; break ;;
  esac
done
[[ $family == "" ]] && have apt-get && family=apt

as_root=""
if (( EUID != 0 )); then
  if have sudo; then as_root="sudo "; else as_root="(as root) "; fi
fi

if [[ $mode == check ]]; then exit 1; fi

# apt_cand <package>: the version apt would install, or nothing.
apt_cand() {
  have apt-cache || return 0
  apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/ && $2 != "(none)" {print $2}'
}

case $family in
  apt)
    pkgs=(); ppa=0
    if (( need_cmake )); then
      cand=$(apt_cand cmake)
      if [[ $cand =~ ^([0-9]+)\.([0-9]+) ]] && ! (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] >= 20) )); then
        echo "This system's cmake package is $cand, older than the 3.20 the build needs."
        echo "Install a newer one with:  pip install cmake   (or the installer from cmake.org), then run make again."
        exit 1
      fi
      pkgs+=(cmake)
    fi
    if (( need_cxx )); then
      gxx=""
      for p in g++-14 g++-13 g++-12; do
        [[ -n $(apt_cand "$p") ]] && { gxx=$p; break; }
      done
      # No package lists fetched yet (a fresh container): go by the release.
      if [[ -z $gxx && -z $(apt_cand g++) ]]; then
        { [[ $os_id == ubuntu && $os_major -ge 24 ]] || [[ $os_id == debian && $os_major -ge 12 ]]; } && gxx=g++
        [[ $os_id == ubuntu && $os_major -eq 22 ]] && gxx=g++-12
      fi
      if [[ -z $gxx ]]; then
        if [[ $os_id == ubuntu ]]; then
          ppa=1; gxx=g++-12
        else
          echo "This system has no g++ 12 or newer in its package archive."
          echo "Use a newer release, or build in the container (docs/docker.md), or install clang 17 or newer."
          exit 1
        fi
      fi
      pkgs+=("$gxx")
    fi
    sudo_pfx=""; [[ -n $as_root && $as_root != "(as root) " ]] && sudo_pfx=sudo
    cmds=()
    if (( ppa )); then
      cmds+=("${sudo_pfx:+$sudo_pfx }apt-get update" "${sudo_pfx:+$sudo_pfx }apt-get install -y software-properties-common"
             "${sudo_pfx:+$sudo_pfx }add-apt-repository -y ppa:ubuntu-toolchain-r/test")
    fi
    cmds+=("${sudo_pfx:+$sudo_pfx }apt-get update" "${sudo_pfx:+$sudo_pfx }apt-get install -y ${pkgs[*]}")
    if (( ppa )) && [[ ${YES:-} != 1 || ${PPA:-} != 1 ]]; then
      echo "Ubuntu $os_ver has no g++ 12 in its main archive. It is in the toolchain test PPA"
      echo "(ppa:ubuntu-toolchain-r/test), which this will not add on its own. The commands:"
      for c in "${cmds[@]}"; do echo "  $c"; done
      echo "  export CXX=g++-12        # then run make again"
      echo "To let make do all of it: make deps YES=1 PPA=1"
      exit 1
    fi
    if [[ -n $as_root && $as_root == "(as root) " ]]; then
      echo "Run, as root:"
      for c in "${cmds[@]}"; do echo "  $c"; done
      exit 1
    fi
    echo "Will run:"
    for c in "${cmds[@]}"; do echo "  $c"; done
    if [[ ${YES:-} != 1 ]]; then
      if [[ ! -t 0 ]]; then
        echo "Not installing: no terminal to ask on. Run the commands above, or: make deps YES=1"
        exit 1
      fi
      read -r -p "Install ${pkgs[*]} now? [y/N] " ans
      case $ans in y|Y|yes|YES) ;; *) echo "Nothing installed. Run the commands above when ready."; exit 1 ;; esac
    fi
    for c in "${cmds[@]}"; do
      echo "+ $c"
      # shellcheck disable=SC2086  # the command text is words by construction
      DEBIAN_FRONTEND=noninteractive $c || { echo "deps: '$c' failed." >&2; exit 1; }
    done
    ;;
  dnf)
    cxx_pkg=gcc-c++
    case $os_id in rhel|rocky|almalinux|centos) cxx_pkg=gcc-toolset-13-gcc-c++ ;; esac
    pk=(); (( need_cmake )) && pk+=(cmake); (( need_cxx )) && pk+=("$cxx_pkg")
    echo "Run:  ${as_root}dnf install -y ${pk[*]} make"
    [[ $cxx_pkg == gcc-toolset-13-gcc-c++ ]] && (( need_cxx )) && echo "then:  source scl_source enable gcc-toolset-13   (in the shell you build from)"
    exit 1 ;;
  pacman)
    echo "Run:  ${as_root}pacman -S --needed cmake gcc make"; exit 1 ;;
  zypper)
    pk=(); (( need_cmake )) && pk+=(cmake); (( need_cxx )) && pk+=(gcc13-c++)
    echo "Run:  ${as_root}zypper install ${pk[*]} make"; exit 1 ;;
  *)
    echo "This system's package manager is not known here. Install cmake 3.20+ and g++ 12+ (or clang++ 17+) with it."
    echo "'make doctor' says what is missing."; exit 1 ;;
esac

# ---- did it work --------------------------------------------------------------
hash -r
if cmake_ok && { cxx_good "$default_cxx" || cxx_good g++-14 || cxx_good g++-13 || cxx_good g++-12 || cxx_good g++; }; then
  echo "Installed. 'make build' is next."
  exit 0
fi
echo "deps: the packages are installed but 'make doctor' still finds a problem." >&2
exit 1
