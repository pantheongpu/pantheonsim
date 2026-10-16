# Which runner a CI job uses

Each heavy job takes its runner from a repository variable (Settings > Secrets and variables > Actions >
Variables). With the variable unset the job runs on GitHub's `ubuntu-24.04`, so a fork or a fresh clone needs
nothing set.

| Variable | Jobs | Needs on the runner |
| --- | --- | --- |
| `BURST_RUNS_ON` | SASS, sanitizers, threads, Ollama on AMD | docker; `sudo apt-get` without a password (the Ollama job installs packages on the host) |
| `CONTAINER_RUNS_ON` | `sass`, `sanitizers`, `threads` only | docker, and nothing else: these three run inside the CI image |
| `ML_RUNS_ON` | ML, external suites, PyTorch sweep | the baked runner image (CUDA 13, PyTorch, ROCm, vLLM) |
| `SIM_RUNS_ON` | simulator jobs | see the workflow |

`CONTAINER_RUNS_ON` is a label list like the others (every label must be on the runner). When it is unset, the three
jobs use `BURST_RUNS_ON`, as before. Give the same label to every runner that should share them (a runner takes any
job whose labels it has, so a pool is just one label on several machines), then
`gh variable set CONTAINER_RUNS_ON --body '["pantheonsim-container"]'`.

Public repositories need a runner group that allows them, and the repository's Settings > Actions > General should
require approval for all outside collaborators, so a fork's pull request runs nothing until a maintainer approves it.
