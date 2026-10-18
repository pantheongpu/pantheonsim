#!/usr/bin/env python3
"""Collect CUPTI traces from a Lambda Cloud GPU (the Lambda counterpart of trace-aws.sh).

    nvidia/tools/trace-lambda.py <instance-type> [--region R] [--wait SECONDS]

Launches one instance, uploads the CUPTI e2e cases and trace-collect-remote.sh,
runs them, copies the traces and binaries back to $VGPU_CLOUD_OUT/<type>/
(default /tmp/vgpu-trace) and terminates the instance, also on error or signal.
Only the instance this run launched is ever terminated: the Lambda account is shared.

Launching reuses the bench drivers' Lambda client (capacity polling, boot-failure
retry, mid-boot teardown). Set VGPU_BENCH_DIR to the directory holding providers.py
and remote.py (default ~/tools/pantheon/bench).
"""
import argparse, os, signal, subprocess, sys, glob

sys.path.insert(0, os.path.expanduser(os.environ.get("VGPU_BENCH_DIR", "~/tools/pantheon/bench")))
import time  # noqa: E402
import requests  # noqa: E402
import providers  # noqa: E402

# This workstation's network drops now and then (DNS failures, resets). The bench client does not
# retry, and a drop inside the boot wait costs a rig, so retry connection errors here.
_api = providers._lambda
def _retrying(method, path, **kw):
    for attempt in range(8):
        try:
            return _api(method, path, **kw)
        except requests.exceptions.ConnectionError as exc:
            print(f"[lambda] network error ({type(exc).__name__}), retry {attempt + 1}/8", flush=True)
            time.sleep(10)
    return _api(method, path, **kw)
providers._lambda = _retrying

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "..", "tests", "e2e")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("type")
    ap.add_argument("--region")
    ap.add_argument("--wait", type=int, default=1800, help="seconds to wait for capacity")
    args = ap.parse_args()
    outdir = os.path.join(os.environ.get("VGPU_CLOUD_OUT", "/tmp/vgpu-trace"), args.type)
    os.makedirs(outdir, exist_ok=True)

    host = None
    def teardown(*_):
        if host is not None:
            print(f"[{args.type}] terminating {host.instance_ref}", flush=True)
            providers.lambda_terminate(host.instance_ref)
    signal.signal(signal.SIGTERM, lambda *a: (teardown(), sys.exit(1)))
    try:
        host = providers.lambda_launch(args.type, f"vgpu-trace-{args.type}", region=args.region,
                                       wait_for_capacity=args.wait)
        host.wait_for_ssh()
        host.ssh("mkdir -p /tmp/tr")
        files = (glob.glob(os.path.join(E2E, "cupti_*.cu")) + glob.glob(os.path.join(E2E, "cupti_*.inc"))
                 + [os.path.join(E2E, "cupti_test_util.h"), os.path.join(HERE, "trace-collect-remote.sh")])
        for f in files:
            host.push(f, "/tmp/tr/")
        env = f"TRACE_CUDA={os.environ.get('TRACE_CUDA', '')}"
        log = host.ssh(f"{env} bash /tmp/tr/trace-collect-remote.sh /tmp/tr", timeout=3000, check=False)
        open(os.path.join(outdir, "..", f"{args.type}.run.log"), "w").write(log.stdout or "")
        host.pull("/tmp/tr/out/*", outdir)
        ok = (os.path.exists(os.path.join(outdir, "trace.txt"))
              and "BUILD FAILED" not in open(os.path.join(outdir, "trace.txt")).read(200)
              and os.path.getsize(os.path.join(outdir, "info.txt")) > 0)
        print(f"[{args.type}] {'collected -> ' + outdir if ok else 'FAILED: see ' + outdir}")
        return 0 if ok else 1
    finally:
        teardown()


if __name__ == "__main__":
    sys.exit(main())
