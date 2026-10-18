#!/usr/bin/env python3
"""Collect CUPTI traces from a Vast.ai GPU (the marketplace counterpart of trace-aws.sh).

    nvidia/tools/trace-vast.py "RTX 4090" [--max-dph 0.9] [--min-cuda 13.0]

Rents the cheapest reliable single-GPU offer whose driver runs CUDA 13.0 (the toolkit the
simulator's expected files come from), runs trace-collect-remote.sh in a CUDA devel container,
copies the traces and binaries to $VGPU_CLOUD_OUT/<model>/ and destroys the instance, also on
error or signal. One instance at a time (the account's limit is two); only this run's instance
is destroyed. Uses the bench drivers' Vast client: VGPU_BENCH_DIR (default ~/tools/pantheon/bench).
"""
import argparse, glob, os, signal, sys, time

sys.path.insert(0, os.path.expanduser(os.environ.get("VGPU_BENCH_DIR", "~/tools/pantheon/bench")))
import vast_sweep as vs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "..", "tests", "e2e")
IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-dph", type=float, default=0.9)
    ap.add_argument("--min-cuda", type=float, default=13.0)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()
    slug = args.model.replace(" ", "").lower()
    outdir = os.path.join(os.environ.get("VGPU_CLOUD_OUT", "/tmp/vgpu-trace"), slug)
    os.makedirs(outdir, exist_ok=True)
    vs.LABEL = "vgpu-trace"
    vs.ensure_ssh_key()

    live = [i for i in (vs.vast("show", "instances") or []) if isinstance(i, dict)]
    if len(live) >= 2:
        print(f"[{slug}] two Vast instances are already live; not renting a third")
        return 2

    offers = vs.drop_unreliable(vs.search_offers([args.model], args.max_dph, args.min_cuda), 0.95)
    offers.sort(key=lambda o: o.get("dph_total") or o.get("dph") or 1e9)
    print(f"[{slug}] {len(offers)} offers; cheapest ${(offers[0].get('dph_total') or 0):.2f}/h" if offers else f"[{slug}] no offers")

    inst_id = None
    def teardown(*_):
        if inst_id is not None:
            print(f"[{slug}] destroying {inst_id}", flush=True)
            vs.destroy_instance(inst_id)
    signal.signal(signal.SIGTERM, lambda *a: (teardown(), sys.exit(1)))
    for offer in offers[:args.tries]:
        try:
            inst_id = vs.create_instance(int(offer["id"]), "", args.image)
            print(f"[{slug}] instance {inst_id} on machine {offer.get('machine_id')} (${offer.get('dph_total', 0):.2f}/h)", flush=True)
            host = vs.wait_ready(inst_id, timeout=1500)
            host.ssh("mkdir -p /tmp/tr")
            files = (glob.glob(os.path.join(E2E, "cupti_*.cu")) + glob.glob(os.path.join(E2E, "cupti_*.inc"))
                     + [os.path.join(E2E, "cupti_test_util.h"), os.path.join(HERE, "trace-collect-remote.sh")])
            for f in files:
                host.push(f, "/tmp/tr/")
            log = host.ssh("bash /tmp/tr/trace-collect-remote.sh /tmp/tr", timeout=3000, check=False)
            open(os.path.join(outdir, "..", f"{slug}.run.log"), "w").write(log.stdout or "")
            host.pull("/tmp/tr/out/*", outdir)
            t = os.path.join(outdir, "trace.txt")
            ok = os.path.exists(t) and "BUILD FAILED" not in open(t).read(200)
            print(f"[{slug}] {'collected -> ' + outdir if ok else 'FAILED: see ' + outdir}")
            return 0 if ok else 1
        except (RuntimeError, TimeoutError) as exc:
            print(f"[{slug}] offer failed: {exc}", flush=True)
        finally:
            teardown()
            inst_id = None
    return 1


if __name__ == "__main__":
    sys.exit(main())
