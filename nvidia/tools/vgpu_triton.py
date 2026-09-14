"""Run Triton kernels on VirtualGPU.

Triton compiles Python to PTX and then shells out to ``ptxas`` to turn that PTX
into a cubin, which it loads with ``cuModuleLoadData``. VirtualGPU interprets
PTX and has no SASS decoder, so the cubin is the one artifact in that pipeline
it cannot use -- the PTX one step earlier is exactly what it wants.

This installs a hook that ends Triton's pipeline at PTX. It uses
``knobs.runtime.add_stages_inspection_hook``, Triton's own documented extension
point for rewriting compilation stages, so nothing here depends on Triton
internals beyond that one name.

Usage::

    import vgpu_triton; vgpu_triton.install()
    # ... ordinary Triton code from here on

or, without editing the program::

    PYTHONSTARTUP=... python -c "import vgpu_triton; vgpu_triton.install(); exec(open('prog.py').read())"

``install()`` is a no-op when Triton is not importable, and is safe to call more
than once.
"""

_INSTALLED = False


def _hook(*args):
    # Triton calls the hook with no arguments to fold it into the compilation
    # cache key -- two different hooks must not share cached kernels -- and with
    # the pipeline when it is time to modify it.
    if not args:
        return ("vgpu-ptx-passthrough", "vgpu-ptx-passthrough")
    _backend, stages, _options, _language, _capability = args
    # The loader reads the image as a C string, so terminate it.
    stages["cubin"] = lambda src, metadata: src.encode() + b"\0"


def available():
    """True if this Triton exposes the stage hook the simulator needs."""
    try:
        from triton import knobs
    except ImportError:
        return False
    return hasattr(knobs, "runtime") and hasattr(knobs.runtime, "add_stages_inspection_hook")


def install():
    """Point Triton's cubin stage at the PTX it already produced.

    Returns True if the hook was installed. Returns False if Triton is not
    installed at all; raises if Triton is present but too old to expose
    ``knobs.runtime.add_stages_inspection_hook``, since that is a version
    problem worth naming rather than a missing optional dependency.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import triton
        from triton import knobs
    except ImportError:
        return False
    if not hasattr(knobs, "runtime") or not hasattr(knobs.runtime, "add_stages_inspection_hook"):
        raise RuntimeError(
            "Triton %s does not expose knobs.runtime.add_stages_inspection_hook; "
            "VirtualGPU needs that hook to stop the pipeline at PTX"
            % getattr(triton, "__version__", "?")
        )
    if knobs.runtime.add_stages_inspection_hook not in (None, _hook):
        raise RuntimeError(
            "another add_stages_inspection_hook is already installed; "
            "VirtualGPU needs this hook to stop Triton's pipeline at PTX"
        )
    knobs.runtime.add_stages_inspection_hook = _hook
    _INSTALLED = True
    return True
