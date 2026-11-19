# Which runner a CI job uses

Each heavy job takes its runner from a repository variable (Settings > Secrets and variables > Actions >
Variables). With the variable unset the job runs on GitHub's `ubuntu-24.04`, so a fork or a fresh clone needs
nothing set.

| Variable | Jobs | Runner it points at |
| --- | --- | --- |
| `BURST_RUNS_ON` | SASS, sanitizers, threads (unless `CONTAINER_RUNS_ON` is set), Ollama on NVIDIA and on AMD | `["self-hosted","pantheongpu"]`: any shared runner (docker; `sudo apt-get` without a password for the Ollama jobs) |
| `CONTAINER_RUNS_ON` | `sass`, `sanitizers`, `threads`, and the image finder they wait for | unset: falls back to `BURST_RUNS_ON` |
| `ML_RUNS_ON` | ML, external suites, PyTorch sweep | `["ml"]`: the one runner with the baked image (CUDA 13, PyTorch, ROCm, vLLM) |
| `SIM_RUNS_ON` | simulator jobs | unset: GitHub-hosted (see the workflow) |

## The shared pool (2026-11-18)

The runners belong to the organisation, not to a repository, and all sit in its Default group, so any repo can use
them. Two labels do the routing, and the rest of a runner's labels are GitHub's defaults:

* `pantheongpu` (with `docker`): the general pool, on `server1-sim`, `server2-1` and `server2-2`. A job that asks for
  `[self-hosted, pantheongpu]` runs on whichever of them is free. Adding capacity is one more runner with that label.
* `ml`: the runner on the pantheonsim.com host, registered **without** the default labels so a plain `self-hosted` job
  can never land next to the live playground. It is the only runner with the baked ML image.

A runner takes any job whose labels it has, so a pool is just one label on several machines. Give a new runner
`pantheongpu`, nothing else. The old per-purpose labels (`pantheonsim-ci`, `pantheonsim-burst`, `pantheonsim-container`,
`server2`) were removed; do not bring them back.

Public repositories need a runner group that allows them, and the repository's Settings > Actions > General should
require approval for all outside collaborators, so a fork's pull request runs nothing until a maintainer approves it.

## Changing a runner's registration

Notes kept from the move of the runners to the organisation.

* **Wait for an idle moment.** Stop a runner only while it has no `Runner.Worker` process, and check again right before
  the stop: a job can start in the second between the check and the stop.
* **Fetch the registration token last.** It lasts one hour, so get it after the wait, not before.
* **Without root.** A runner can change its registration with nothing but `systemctl stop|start` of its own unit: stop
  the unit, delete the old registration through the API, remove `.runner*` and `.credentials*` in the runner's
  directory, run `config.sh --replace` with the new token and labels, start the unit. The unit keeps its old name,
  which is cosmetic. `config.sh remove` needs root and is not required.
* **"Still counts a job".** GitHub refuses to delete a runner it thinks is running a job. A lost job clears after a
  minute or two: retry every 20 seconds.
* **Keep what is beside the registration.** The runner's `.env` (memory-gate and clean-up hooks), its `_work` folder and
  the systemd drop-in (`Slice=ci.slice`, `Restart=always`, `RestartSec=30`) are not part of the registration. Copy the
  `.env` and write the drop-in again when a runner is added or its service is reinstalled.
