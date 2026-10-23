# Security

## Reporting a vulnerability

Please don't report security problems in a public issue.

Report them privately instead, in either of these ways:

- On GitHub: the **Security** tab of this repository, then **Report a
  vulnerability**.
- By email: saqibkhan@pantheonsim.com.

Say what you found, how to reproduce it, and which commit you tested. We will
acknowledge the report within 7 days, keep you updated while we work on a fix,
and credit you in the fix unless you'd rather not be named.

## What's in scope

- The simulator itself: the parsers for the binaries it reads (nvcc fatbins,
  PTX and AMD code objects), the interpreters, and the CLI.
- The libraries it provides in place of the vendor ones (the CUDA runtime and
  driver, the HIP runtime, and the math and communication libraries).
- The GitHub Action in this repository.

Problems with the playground at pantheonsim.com can be reported the same way.

## What PantheonSim does not protect against

**PantheonSim is not a sandbox.** A program you run through it runs as an
ordinary process on your machine, with your user's permissions. Simulating a GPU
does not isolate the program's host code, and a bug in the simulator could let
a kernel affect the rest of the process.
Treat running an untrusted binary through PantheonSim exactly like running it
directly: do it in a container or a virtual machine.

The opt-in JIT (`VGPU_JIT=1`, off by default) runs your C++ compiler on text it generates from the kernel's instruction words, and loads the
result into the process. The text holds only numbers and fixed templates, never a string taken from the program being simulated. The cache
directory is created `0700`, must belong to you and be writable by no one else, and an object is loaded only after its checksum and its
instruction words match; but the objects are native code that runs with your permissions, so do not point `VGPU_JIT_CACHE` at a directory
other users can write to.

## Supported versions

Fixes land on `main`. The `pantheongpu/setup-pantheonsim` action pins a commit
of this repository, and that pin is moved forward when a security fix lands.
