#!/usr/bin/env python3
"""Every runtime entry point that puts work on a stream has to know about capture.

A stream that is capturing does not run what it is given: it records it, and the
work runs when the graph does. An entry point that forgets this runs the work at
capture time instead, and every replay of the graph silently leaves it out --
the answers are wrong and nothing fails. That happened here to 2D, symbol and
peer copies, host functions, stream-ordered allocation, graph launches and
events, one after another.

So this reads every exported cuda* function in the runtime shim that takes a
cudaStream_t, and requires that it either consults capture state or is listed
below with the reason it does not need to. A new stream function that does
neither fails this check before it can ship.
"""
import pathlib
import re
import sys

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "nvidia" / "src" / "runtime_api.cpp"

# Calls that mean a function takes part in capture: it records, refuses, or
# answers about it.
CAPTURE_AWARE = re.compile(
    r"\b(capture_active|capture_record|capture_refuse|capture_host_fn|capture_malloc|"
    r"capture_free|capture_wait|capture_event_node|capture_position|capture_target|"
    r"stream_capture|capture_info|update_capture_deps|vgpu_record_\w+|vgpu_drop_capture|"
    r"launch_kernel_impl)\s*\(")

# Stream functions that need no capture handling, each with why. Adding a name
# here is a claim about CUDA's semantics, so it needs a reason that holds.
EXEMPT = {
    # They ask about a stream or change a property of it; nothing is enqueued.
    "cudaStreamGetFlags": "a query",
    "cudaStreamGetPriority": "a query",
    "cudaStreamGetDevice": "a query",
    "cudaStreamGetId": "a query",
    "cudaStreamGetAttribute": "a query",
    "cudaStreamSetAttribute": "a stream property, not work",
    "cudaStreamCopyAttributes": "stream properties, not work",
    "cudaStreamIsCapturing": "reports capture state (through capture_target)",
    # Managed memory has one physical copy here, so a prefetch moves nothing and
    # there is nothing for a replay to repeat.
    "cudaMemPrefetchAsync": "managed memory has one copy; nothing to replay",
    # There is no device to stage a graph onto; uploading does nothing either way.
    "cudaGraphUpload": "nothing to upload, captured or not",
}


def exported_stream_functions(text):
    """Yields (name, body) for each exported cuda* function with a stream parameter."""
    for m in re.finditer(r"VGPU_EXPORT\s+cudaError_t\s+(cuda\w+)\s*\(([^)]*)\)\s*\{", text):
        name, params = m.group(1), m.group(2)
        # A stream taken by value is one work goes onto; a cudaStream_t* is a
        # stream being handed out (cudaStreamCreate), which enqueues nothing.
        if not re.search(r"cudaStream_t(?!\s*\*)", params):
            continue
        depth, i = 1, m.end()
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        yield name, text[m.end():i]


def main():
    text = SOURCE.read_text()
    bodies = dict(exported_stream_functions(text))
    missing = []
    for name, body in sorted(bodies.items()):
        if name in EXEMPT or CAPTURE_AWARE.search(body):
            continue
        # A thin wrapper that forwards to another stream function takes part if
        # the one it forwards to does.
        forwards = [f for f in re.findall(r"\b(cuda\w+)\s*\(", body) if f in bodies and f != name]
        if any(f in EXEMPT or CAPTURE_AWARE.search(bodies[f]) for f in forwards):
            continue
        missing.append(name)
    stale = sorted(n for n in EXEMPT if n not in bodies)
    for name in missing:
        print(f"FAIL {name} takes a stream but never consults capture state: on a capturing "
              f"stream it would run now and be missing from every replay. Record it, refuse it "
              f"(capture_refuse), or add it to EXEMPT with the reason it needs neither.")
    for name in stale:
        print(f"FAIL EXEMPT names {name}, which is no longer an exported stream function")
    if missing or stale:
        return 1
    print(f"capture coverage: {len(bodies)} stream functions, "
          f"{len(bodies) - len(EXEMPT)} capture-aware, {len(EXEMPT)} exempt with a reason: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
