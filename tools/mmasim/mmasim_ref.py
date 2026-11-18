#!/usr/bin/env python3
"""Run the REAL MMA-Sim (github.com/microsoft/MMA-Sim, pip install mmasim; needs torch >= 2.9) over the vectors that
`mmasim_xcheck export <dir>` wrote, and write <dir>/<id>.out for `mmasim_xcheck compare <dir>`.

  mmasim_xcheck export /tmp/vec --n 20000
  python3 tools/mmasim/mmasim_ref.py /tmp/vec
  mmasim_xcheck compare /tmp/vec

Untested in this repository's history: it was written without torch on the machine where the harness could be run
(docs/numerics/mmasim-crosscheck.md says so). Importing mmasim compiles a small C++ extension (torch.utils.cpp_extension), so run it where
that is acceptable. Each vector is one output element: a [n, K], b [n, K], c [n] go to the model's `dpa`, which is what its `__call__`
does per output element (arithmetic/fdpa.py). No GPU is used.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from mmasim.amd import MFMA
from mmasim.nv_ptx import MMA, TCGen05MMA, WGMMA

OPS = {"mma": MMA, "wgmma": WGMMA, "tcgen05": TCGen05MMA, "mfma": MFMA}
DT = {"f16": torch.float16, "bf16": torch.bfloat16, "tf32": torch.float32, "f32": torch.float32}


def main(d: Path) -> None:
    for line in (d / "manifest.tsv").read_text().splitlines():
        gid, side, arch, api, instr, K, n, fmt, acc = line.split("\t")
        K, n = int(K), int(n)
        words = np.fromfile(d / f"{gid}.in", dtype=np.uint32).reshape(n, 2 * K + 1)
        f = torch.from_numpy(words.view(np.float32).copy())  # values are exact in their format
        a, b, c = f[:, :K], f[:, K : 2 * K], f[:, 2 * K]
        a, b = a.to(DT[fmt]), b.to(DT[fmt])  # tf32 stays float32 (MMA-Sim truncates it itself; the vectors already are)
        c = c.to(torch.float16 if acc == "f16" else torch.float32)
        op = OPS[api](arch, instr)
        with torch.no_grad():
            r = op.dpa(a, b, c)
        if acc == "f16":
            out = (r.view(torch.int16).numpy().astype(np.int64) & 0xFFFF).astype(np.uint32)
        else:
            out = r.view(torch.int32).numpy().astype(np.int64).astype(np.uint32)
        out.tofile(d / f"{gid}.out")
        print(f"{arch} {api} {instr}: {n} vectors", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
