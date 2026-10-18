#!/usr/bin/env python3
"""Collect CUPTI traces from a RunPod GPU (the RunPod counterpart of trace-aws.sh).

    nvidia/tools/trace-runpod.py "NVIDIA H100 80GB HBM3" [--max-dph 8]
    nvidia/tools/trace-runpod.py --list          # the GPU names RunPod quotes

Rents one pod in a CUDA 13.0 devel container, runs trace-collect-remote.sh, copies the traces and
binaries to $VGPU_CLOUD_OUT/<name>/ (default /tmp/vgpu-trace) and deletes the pod, also on error or
signal. At most two pods of ours at a time (the account's rule); only this run's pod is deleted.
Uses the bench drivers' RunPod client: VGPU_BENCH_DIR (default ~/tools/pantheon/bench).
"""
import argparse, glob, os, re, signal, sys

sys.path.insert(0, os.path.expanduser(os.environ.get("VGPU_BENCH_DIR", "~/tools/pantheon/bench")))
import runpod_sweep as rp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "..", "tests", "e2e")
IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
# The pod's start command: sshd for us to log in with, then stay up (a container whose PID 1 exits
# is a pod that stops).
START = """set -u
export DEBIAN_FRONTEND=noninteractive
mkdir -p /run/sshd /root/.ssh; chmod 700 /root/.ssh
printf "%s\\n" "${PUBLIC_KEY:-}" >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys
(apt-get update -qq && apt-get install -y -qq --no-install-recommends openssh-server) > /tmp/sshd-install.log 2>&1
/usr/sbin/sshd -D > /tmp/sshd.log 2>&1 &
sleep infinity
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-dph", type=float, default=8.0)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--secure", action="store_true", help="try the secure cloud first")
    ap.add_argument("--tries", type=int, default=4, help="pods to try before giving up")
    args = ap.parse_args()
    if args.list:
        for g in rp.gpu_catalogue():
            print(f"{g['usd_per_hour']:6.2f}  {g['vram_gb']:>4} GB  {g['name']}")
        return 0
    if not args.gpu:
        ap.error("a GPU name, or --list")
    slug = re.sub(r"[^a-z0-9]+", "", args.gpu.lower().replace("nvidia", "").replace("geforce", ""))
    outdir = os.path.join(os.environ.get("VGPU_CLOUD_OUT", "/tmp/vgpu-trace"), f"runpod-{slug}")
    os.makedirs(outdir, exist_ok=True)

    pods = rp.live_pods()
    if pods is None or len(pods) >= 2:
        print(f"[{slug}] {'cannot read the pod list' if pods is None else 'two pods of ours are live'}; not renting")
        return 2
    gpus = rp.select([args.gpu], args.max_dph)
    if not gpus:
        print(f"[{slug}] RunPod quotes no '{args.gpu}' at or under ${args.max_dph}/h (see --list)")
        return 2

    def teardown(pid):
        if pid is not None:
            print(f"[{slug}] deleting pod {pid}", flush=True)
            rp.delete_pod(pid)

    pod_id = None
    signal.signal(signal.SIGTERM, lambda *a: (teardown(pod_id), sys.exit(1)))
    for attempt in range(1, args.tries + 1):
        pod_id = None
        try:
            pod = rp.create_pod(gpus[0], START, args.image, 30, prefer_secure=args.secure)
            pod_id = str(pod["id"])
            print(f"[{slug}] pod {pod_id} at ${float(pod.get('costPerHr') or gpus[0]['usd_per_hour']):.2f}/h "
                  f"({pod.get('_placement', '?')}), attempt {attempt}", flush=True)
            host = rp.wait_ready(pod_id, gpus[0]["name"])
            # The image's toolkit is CUDA 13.0; a host whose driver is older cannot run it, and the
            # placement pin does not always keep such hosts away.
            smi = host.ssh("nvidia-smi", check=False).stdout or ""
            m = re.search(r"CUDA Version: (\d+)\.(\d+)", smi)
            if not m or int(m.group(1)) < 13:
                print(f"[{slug}] driver supports CUDA {'.'.join(m.groups()) if m else '?'}, need 13: another host", flush=True)
                continue
            host.ssh("mkdir -p /tmp/tr")
            files = (glob.glob(os.path.join(E2E, "cupti_*.cu")) + glob.glob(os.path.join(E2E, "cupti_*.inc"))
                     + [os.path.join(E2E, "cupti_test_util.h"), os.path.join(HERE, "trace-collect-remote.sh")])
            for f in files:
                host.push(f, "/tmp/tr/")
            log = host.ssh(f"TRACE_ONLY='{os.environ.get('TRACE_ONLY', '')}' bash /tmp/tr/trace-collect-remote.sh /tmp/tr",
                           timeout=3000, check=False)
            open(os.path.join(outdir, "..", f"runpod-{slug}.run.log"), "w").write(log.stdout or "")
            host.pull("/tmp/tr/out/*", outdir)
            t = os.path.join(outdir, "trace.txt")
            ok = os.path.exists(t) and "BUILD FAILED" not in open(t).read(200)
            print(f"[{slug}] {'collected -> ' + outdir if ok else 'FAILED: see ' + outdir}")
            return 0 if ok else 1
        except (RuntimeError, TimeoutError) as exc:
            print(f"[{slug}] failed: {exc}", flush=True)
        finally:
            teardown(pod_id)
    return 1


if __name__ == "__main__":
    sys.exit(main())
