# Running PantheonSim on CI systems other than GitHub Actions

`scripts/ci-setup.sh` does what the GitHub Action does (install a compiler,
build the simulator or reuse a build, put the simulated GPUs on `PATH`) with no
dependency on GitHub, so any CI system that can run a shell script on an Ubuntu
x86_64 machine or container can use it. See [github-action.md](github-action.md)
for what the simulator gives a job and for the notes on `library-path`.

## What was tested

Only this: the script itself, run in a plain `ubuntu:24.04` Docker container as
root with no `sudo`, with `PSIM_GPU=nvidia/t4 PSIM_CUDA_TOOLKIT=12.6`, then
`nvcc` and a vector add under `vgpu run`, then a second run that reused the
build. The workflow `.github/workflows/ci-setup-script.yml` runs the same
sequence on a GitHub runner.

**Every CI snippet below is untested.** None was run on its own platform. They
use the script the way each system's documentation describes, but expect to
adjust them. Also untested: the AMD/ROCm path of the script, Ubuntu 22.04, the
`sudo` path, and `PSIM_CUDA_TOOLKIT=apt`.

## The script

```sh
git clone https://github.com/pantheongpu/pantheonsim.git
git -C pantheonsim checkout <commit>     # pin one, so the script does not change under you
PSIM_GPU=nvidia/h100 PSIM_COUNT=2 PSIM_CUDA_TOOLKIT=12.6 pantheonsim/scripts/ci-setup.sh
. ./.pantheonsim-build.env              # see "The env file" below
nvcc -arch=compute_90 my_test.cu -o my_test && vgpu run ./my_test
```

| Variable | Default | Same as the action's input |
| --- | --- | --- |
| `PSIM_GPU` | `nvidia/t4` | `gpu` |
| `PSIM_COUNT` | `1` | `count` |
| `PSIM_CUDA_TOOLKIT` | `apt` on Ubuntu 24.04, `12.6` on older Ubuntu, `none` for AMD | `cuda-toolkit` |
| `PSIM_ROCM` | `7.1` for AMD, `none` for NVIDIA | `rocm` |
| `PSIM_LIBRARY_PATH` | `true` | `library-path` |
| `PSIM_BUILD_DIR` | `$PWD/.pantheonsim-build` | the cached build directory |
| `PSIM_ENV_FILE` | `$PSIM_BUILD_DIR.env` | (none) |

Requirements: Linux x86_64, Ubuntu 22.04 or 24.04 (use an `ubuntu:24.04` image
if your runner is something else), root or `sudo`, network access to the Ubuntu
mirrors, `developer.download.nvidia.com` (when `PSIM_CUDA_TOOLKIT` is a
version), `repo.radeon.com` (AMD) and PyPI (cuDNN headers fetched while
configuring). No GPU and no privileged container. The toolkit is installed from
Ubuntu's or the vendor's own repository on your runner, as the action does;
nothing proprietary is shipped with PantheonSim.

### The env file

A script cannot change the environment of the shell that started it, and most
CI systems start a new shell for every step. So the script writes `PATH`,
`VGPU_*`, `LD_LIBRARY_PATH` and the rest to the env file
(`$PSIM_BUILD_DIR.env`, next to the build directory) and you source it in each
step that needs the GPUs: `. "$PSIM_ENV_FILE"`. The file holds absolute paths, so
source it in the job that ran the script, or run the script again (it is fast,
see below).

### Caching

The first run builds the simulator: about 7.5 minutes on a 20-core
machine (measured, CUDA 12.6 install included), longer on a small runner. Cache the `PSIM_BUILD_DIR` directory
between runs. The script records a key in it (the sources, the OS release and
the nvcc version, the same as the action's cache key) and rebuilds when it does
not match, so a stale cache is never used; a matching one makes a re-run take
seconds (5 s measured). It also prints the key (`pantheonsim: cache key ...`), if your CI
system can key a cache on a string you compute first. On a long-lived runner
(Jenkins, a self-hosted Buildkite agent) just keep the directory.

## GitLab CI (untested)

```yaml
pantheonsim:
  image: ubuntu:24.04
  variables:
    PSIM_GPU: nvidia/t4
    PSIM_CUDA_TOOLKIT: "12.6"
    PSIM_BUILD_DIR: $CI_PROJECT_DIR/.pantheonsim-build   # the cache must be inside the project
  cache:
    key: pantheonsim-$PSIM_GPU-$PSIM_CUDA_TOOLKIT
    paths: [.pantheonsim-build/]
  before_script:
    - apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates
    - git clone https://github.com/pantheongpu/pantheonsim.git /opt/pantheonsim
    - git -C /opt/pantheonsim checkout <commit>
    - /opt/pantheonsim/scripts/ci-setup.sh
    - . "$CI_PROJECT_DIR/.pantheonsim-build.env"
  script:
    - nvcc -arch=compute_75 my_test.cu -o my_test
    - vgpu run ./my_test
```

GitLab's cache is per runner unless the runners share a cache backend, so the
build may be repeated; the script handles a missing cache.

## CircleCI (untested)

```yaml
version: 2.1
jobs:
  cuda-tests:
    docker:
      - image: ubuntu:24.04
    resource_class: large
    environment:
      PSIM_GPU: nvidia/t4
      PSIM_CUDA_TOOLKIT: "12.6"
      PSIM_BUILD_DIR: /root/pantheonsim-build
    steps:
      - run: apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates openssh-client
      - checkout
      - run: git clone https://github.com/pantheongpu/pantheonsim.git /opt/pantheonsim && git -C /opt/pantheonsim checkout <commit>
      - restore_cache:
          keys: [pantheonsim-v1-{{ checksum "/opt/pantheonsim/scripts/ci-setup.sh" }}]
      - run: /opt/pantheonsim/scripts/ci-setup.sh
      - save_cache:
          key: pantheonsim-v1-{{ checksum "/opt/pantheonsim/scripts/ci-setup.sh" }}
          paths: [/root/pantheonsim-build]
      - run: |
          . /root/pantheonsim-build.env
          nvcc -arch=compute_75 my_test.cu -o my_test
          vgpu run ./my_test
workflows:
  test:
    jobs: [cuda-tests]
```

## Forgejo and Codeberg (untested)

Forgejo Actions runs GitHub-style workflows, and the existing action may work
by its full URL. Nobody has run it there:

```yaml
jobs:
  cuda-tests:
    runs-on: docker          # a runner label whose image is Ubuntu, see below
    container: ubuntu:24.04
    steps:
      - uses: https://github.com/actions/checkout@v4
      - uses: https://github.com/pantheongpu/pantheonsim@main
        with:
          gpu: nvidia/t4
      - run: |
          nvcc -arch=compute_75 my_test.cu -o my_test
          vgpu run ./my_test
```

Caveats, all unverified: the action's cache step needs the runner's cache
server enabled, and the action's own `uses:` of `actions/cache@v4` resolves
against your instance's default action host rather than GitHub. The default
runner image is Debian, and both the action and the script refuse anything but
Ubuntu, so use `container: ubuntu:24.04` or a label mapped to it. If the action
does not work, call the script instead, as in the shell steps above; a shared
Codeberg runner is small, so expect the first build to be slow.

## Jenkins (untested)

A declarative pipeline on an Ubuntu agent, or in a container. On a long-lived
agent the build directory simply stays between runs:

```groovy
pipeline {
  agent { docker { image 'ubuntu:24.04'; args '-u root' } }
  environment {
    PSIM_GPU = 'nvidia/t4'
    PSIM_CUDA_TOOLKIT = '12.6'
    PSIM_BUILD_DIR = "${env.WORKSPACE}/.pantheonsim-build"
  }
  stages {
    stage('Test') {
      steps {
        sh '''
          apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates
          git clone https://github.com/pantheongpu/pantheonsim.git /opt/pantheonsim
          git -C /opt/pantheonsim checkout <commit>
          /opt/pantheonsim/scripts/ci-setup.sh
          . "$PSIM_BUILD_DIR.env"
          nvcc -arch=compute_75 my_test.cu -o my_test
          vgpu run ./my_test
        '''
      }
    }
  }
}
```

A Docker-backed agent starts empty each time, so the build repeats unless the
workspace or a volume persists.

## Buildkite (untested)

```yaml
steps:
  - label: "CUDA tests"
    command: |
      git clone https://github.com/pantheongpu/pantheonsim.git /opt/pantheonsim
      git -C /opt/pantheonsim checkout <commit>
      /opt/pantheonsim/scripts/ci-setup.sh
      . "$$PSIM_BUILD_DIR.env"
      nvcc -arch=compute_75 my_test.cu -o my_test
      vgpu run ./my_test
    env:
      PSIM_GPU: nvidia/t4
      PSIM_CUDA_TOOLKIT: "12.6"
      PSIM_BUILD_DIR: /var/lib/buildkite-agent/pantheonsim-build   # kept between builds on a self-hosted agent
    plugins:
      - docker#v5.11.0:
          image: ubuntu:24.04
          user: root
          volumes: ["/var/lib/buildkite-agent:/var/lib/buildkite-agent"]
```

(`$$` stops Buildkite interpolating the variable before the shell sees it. The
`docker` plugin version is only an example; use a current one. The image needs
`git`, so add an `apt-get install git ca-certificates` line first.)
