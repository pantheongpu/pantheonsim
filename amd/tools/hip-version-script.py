#!/usr/bin/env python3
"""Writes amd/src/libamdhip64.map: the symbol versions VirtualGPU's libamdhip64
gives each HIP function, taken from a real ROCm libamdhip64.

A program built by hipcc binds each call to a version as well as a name, and the
loader refuses a library that defines the name under any other version, so the
mapping has to be the real library's. Only names are read from it.

  amd/tools/hip-version-script.py <real libamdhip64.so> <our hip_api.cpp.o> [out]

Give it the object file (build/CMakeFiles/vgpuhip.dir/amd/src/hip_api.cpp.o),
not the library: a function added since the map was last written is not in
the library at all, since only what the map exports survives the link.
"""
import subprocess
import sys

ORDER = ['hip_4.2', 'hip_4.3', 'hip_4.4', 'hip_4.5', 'hip_5.0', 'hip_5.1', 'hip_5.2', 'hip_5.3', 'hip_5.5',
         'hip_5.6', 'hip_6.0', 'hip_6.1', 'hip_6.2', 'hip_6.4', 'hip_6.5']

HEADER = '''/* The symbol versions libamdhip64 defines, as ROCm 7.1's own library defines them.
 *
 * A program built by hipcc binds each HIP call to a version as well as a name --
 * hipMalloc@hip_4.2, hipGetDevicePropertiesR0600@hip_6.0 -- and the loader refuses
 * a library that defines the name under any other. So each function here sits in
 * the version the real library gives it. The whole chain of versions is declared
 * even where no function here belongs to one, so a program asking for any of them
 * finds it. Everything that is not HIP's own API stays local to the library.
 *
 * Regenerate with amd/tools/hip-version-script.py when a function is added. */
'''

# VirtualGPU's own: what its librocprofiler-sdk attaches through
# (vgpu/hip_profiler.hpp). No HIP program asks for these.
PRIVATE = '''VGPU_PRIVATE {
  global:
    vgpu_hip_profiler_attach;
    vgpu_hip_profiler_device;
};'''


def versions(real):
    out = subprocess.run(['readelf', '--dyn-syms', '--wide', real], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 8 and '@' in f[7]:
            name, _, ver = f[7].partition('@')
            found.setdefault(name, ver.lstrip('@'))
    return found


def ours(lib, known):
    # The whole symbol table, not the dynamic one: a function added since the
    # map was last written is not exported yet, because the map is what
    # exports it. Such a function is one the real library defines; a local
    # symbol it does not (a compiler's .cold part) stays local.
    out = subprocess.run(['nm', '--defined-only', lib], capture_output=True, text=True).stdout
    names = set()
    for f in (l.split() for l in out.splitlines()):
        if len(f) != 3 or f[1] not in 'Tt':
            continue
        name = f[2].split('@')[0]
        if (name.startswith('hip') or name.startswith('__hip')) and (f[1] == 'T' or name in known):
            names.add(name)
    return sorted(names)


def main():
    real, lib = sys.argv[1], sys.argv[2]
    dest = sys.argv[3] if len(sys.argv) > 3 else 'amd/src/libamdhip64.map'
    known = versions(real)
    nodes = {k: [] for k in ORDER}
    for name in ours(lib, known):
        nodes[known.get(name, 'hip_4.2')].append(name)
    text = [HEADER]
    prev = None
    for k in ORDER:
        body = ''.join(f'    {s};\n' for s in sorted(nodes[k]))
        glob = '  global:\n' + body if body else ''
        loc = '  local: *;\n' if k == 'hip_4.2' else ''
        text.append(f'{k} {{\n{glob}{loc}}}' + (f' {prev};' if prev else ';'))
        prev = k
    text.append(PRIVATE)
    open(dest, 'w').write('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
