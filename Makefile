# PantheonSim front door: thin wrappers around scripts/ and build/vgpu.
#
#   make              install what a build is missing, then build (make deps build)
#   make help         this help
#   make deps         install the missing build tools (asks first; YES=1 to skip the question)
#   make doctor       check the tools a build needs
#   make build        build the simulator (build/vgpu), no installs (half the CPUs; JOBS=N)
#   make demo         run a built-in kernel on the simulated GPU
#   make run APP=./my_app ARGS="..."
#
# The settings (GPU model, how many, memory, checks) live in pantheonsim.yaml,
# which `make init` writes; docs/config.md explains it. Nothing here duplicates
# CMake: `make build` is scripts/build.sh and every other target is one `vgpu`
# command, echoed before it runs so you can copy it.
#
# Variables given on the make command line (GPU=nvidia/t4 COUNT=2 ...) become
# the matching vgpu flags and beat the settings file. Only the command line
# counts, so CUDA_VERSION or COUNT left in your shell's environment do nothing.

VGPU ?= build/vgpu
NICE ?= nice -n 10

# Simulator threads for run, shell and demo: half the CPUs unless VGPU_THREADS is
# set or the settings file has `threads:` (scripts/half-threads.sh). THREADS=N beats it.
ifndef VGPU_THREADS
DEFAULT_THREADS := $(shell ./scripts/half-threads.sh $(if $(filter command line,$(origin CONFIG)),'$(CONFIG)') 2>/dev/null)
ifneq ($(DEFAULT_THREADS),)
export VGPU_THREADS := $(DEFAULT_THREADS)
endif
endif

.DEFAULT_GOAL := all
.NOTPARALLEL:

# --- variables -> flags -------------------------------------------------------
# on_cmdline: the variable was given as `make VAR=value`, not inherited from the environment.
on_cmdline = $(filter command line,$(origin $(1)))
# flag: VAR=value becomes `FLAG 'value'`.
flag = $(if $(call on_cmdline,$(1)),$(2) '$($(1))')
# switch: VAR=1 becomes `FLAG`; VAR=0, no or false does not.
switch = $(if $(call on_cmdline,$(1)),$(if $(filter-out 0 no false,$($(1))),$(2)))

CFG_FLAG    := $(call flag,CONFIG,--config)
SIM_FLAGS   := $(call flag,GPU,--gpu) $(call flag,COUNT,--count) $(call flag,VRAM_MB,--vram-mb) \
               $(call flag,THREADS,--threads) $(call flag,MAX_STEPS,--max-steps) \
               $(call switch,STRICT,--strict) $(call switch,RACE,--race) \
               $(call switch,COUNTERS,--counters) $(call switch,QUIET,--quiet) $(call switch,TRACE,--trace)
RUN_FLAGS   := $(SIM_FLAGS) $(call switch,PRELOAD,--preload)
SHELL_FLAGS := $(call flag,GPU,--gpu) $(call flag,COUNT,--count) $(call flag,VRAM_MB,--vram-mb) \
               $(call flag,CUDA_VERSION,--cuda) $(call flag,ROCM_VERSION,--rocm) \
               $(call flag,DRIVER_VERSION,--driver) $(call flag,SHELL_OS,--os) \
               $(if $(call on_cmdline,ASK),,-y)
DEMO_FLAGS  := $(call flag,GPU,--gpu) $(call flag,N,-n)
# JOBS=N forces the build's parallel jobs (CI_JOBS=N); without it the build uses half the CPUs.
BUILD_ENV   := $(if $(call on_cmdline,JOBS),CI_JOBS='$(JOBS)' )
INIT_ARGS   := $(if $(call on_cmdline,CONFIG),'$(CONFIG)') $(call switch,FORCE,--force)

# run and demo build first when there is no build yet.
BUILD_FIRST := $(if $(wildcard $(VGPU)),,build)

define HELP_TEXT
PantheonSim: run CUDA and HIP programs on simulated GPUs, no GPU needed.

First time:   make        (installs missing build tools after asking, then builds)
              ./run       (starts a simulated machine; settings in pantheonsim.yaml)
Your program: make run APP=./my_app ARGS="--size 1024"

Targets
  make                 make deps, then make build, then say what to type next
  make deps            install cmake and a C++20 compiler if missing (asks; YES=1 to skip the
                       question; PPA=1 as well to add the g++ 13 PPA on Ubuntu 22.04)
  make doctor          check cmake, a C++20 compiler, memory; says what to install
  make build           build the simulator into build/ (low priority; NICE= to change); installs nothing.
                       Uses half the CPUs; JOBS=N sets the number of parallel jobs
  make demo            run a built-in vectorAdd kernel end to end (builds first if needed)
  make list-gpus       the GPU models you can simulate
  make init            write a commented pantheonsim.yaml (FORCE=1 to overwrite)
  make config          show every setting in effect and where it came from
  make shell           a simulated machine: nvidia-smi, rocm-smi, lspci (ASK=1 to be asked)
  make run APP=...     run APP on the simulated GPU (builds first if needed)
  make test            build and run the test suite
  make clean           delete build/

Settings come from pantheonsim.yaml (see docs/config.md). The simulator runs on half the
CPUs unless the file has threads: or VGPU_THREADS is set; THREADS=N overrides both. On the make command line
these override it, for run, config and shell:
  GPU=nvidia/h100  COUNT=8  VRAM_MB=16384  THREADS=4  MAX_STEPS=0
  STRICT=1  RACE=1  COUNTERS=1  QUIET=1  TRACE=1  CONFIG=path/to/file.yaml
run only:   APP=./my_app (or a command such as python)  ARGS="..."  PRELOAD=1 (for python)
demo only:  GPU=...  N=65536
shell only: CUDA_VERSION=13.0  ROCM_VERSION=6.4.0  DRIVER_VERSION=580.65.06  SHELL_OS=ubuntu:24.04

Examples
  make run APP=./vecadd GPU=nvidia/t4
  make run APP=python ARGS="train.py" PRELOAD=1 GPU=amd/mi300x
  make config COUNT=4
endef
export HELP_TEXT

.PHONY: all help deps doctor build demo list-gpus init config shell run test clean need-vgpu check-app

# The default goal: the tools, then the build, then what to do next. A compiler that
# deps just installed (g++-13 on Ubuntu 22.04) is handed to the build through CXX;
# `make build` itself is still exactly scripts/build.sh.
all: deps
	cxx="$$(./scripts/deps.sh --cxx)"; if [ -n "$$cxx" ]; then export CXX="$$cxx"; fi; $(BUILD_ENV)$(NICE) ./scripts/build.sh
	@printf '%s\n' 'Built. Next:' \
	  '  ./run            starts a simulated GPU machine and takes you inside it' \
	  '  its settings     pantheonsim.yaml in this directory (./run writes one); ./run my.yaml for another'

help:
	@printf '%s\n' "$$HELP_TEXT"

deps:
	./scripts/deps.sh

doctor:
	./scripts/doctor.sh

build:
	$(BUILD_ENV)$(NICE) ./scripts/build.sh

demo: $(BUILD_FIRST)
	$(VGPU) $(CFG_FLAG) demo vectoradd $(DEMO_FLAGS)

list-gpus: need-vgpu
	$(VGPU) list-gpus

init: need-vgpu
	$(VGPU) config init $(INIT_ARGS)

config: need-vgpu
	$(VGPU) $(CFG_FLAG) config show $(SIM_FLAGS)

shell: need-vgpu
	$(VGPU) $(CFG_FLAG) shell $(SHELL_FLAGS) $(ARGS)

run: check-app $(BUILD_FIRST)
	$(VGPU) $(CFG_FLAG) run $(RUN_FLAGS) $(APP) $(ARGS)

test:
	$(NICE) ./scripts/test.sh

clean:
	rm -rf build

# Not for typing. The recipes are not echoed (@); the message is the output.
need-vgpu:
	@test -x '$(VGPU)' || { echo "make: $(VGPU) is not built yet. Run 'make build' first (a few minutes, 'make doctor' checks the tools)." >&2; exit 1; }

check-app:
	@if [ -z '$(APP)' ]; then \
	  echo "make run: APP is empty. Say what to run: make run APP=./my_app ARGS=\"...\"" >&2; exit 2; fi; \
	case '$(APP)' in \
	  */*) [ -e '$(APP)' ] || { echo "make run: '$(APP)' does not exist. Build your program first (nvcc -cudart shared my_app.cu -o my_app, or hipcc), then make run APP=./my_app" >&2; exit 2; } ;; \
	  *) command -v '$(APP)' >/dev/null 2>&1 || { echo "make run: '$(APP)' is not a command on PATH. For a program in this directory write APP=./$(APP)" >&2; exit 2; } ;; \
	esac
