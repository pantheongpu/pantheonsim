#!/usr/bin/env python3
"""Writes amd/src/libamdhip64.map: the symbol versions VirtualGPU's libamdhip64
gives each HIP function, taken from a real ROCm libamdhip64.

A program built by hipcc binds each call to a version as well as a name, and the
loader refuses a library that defines the name under any other version, so the
mapping has to be the real library's. Only names are read from it.

  amd/tools/hip-version-script.py <real libamdhip64.so> <our libamdhip64.so> [out]
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


def versions(real):
    out = subprocess.run(['readelf', '--dyn-syms', '--wide', real], capture_output=True, text=True).stdout
    found = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 8 and '@' in f[7]:
            name, _, ver = f[7].partition('@')
            found.setdefault(name, ver.lstrip('@'))
    return found


def ours(lib):
    out = subprocess.run(['nm', '-D', '--defined-only', lib], capture_output=True, text=True).stdout
    return sorted({f[2].split('@')[0] for f in (l.split() for l in out.splitlines())
                   if len(f) == 3 and f[1] == 'T' and (f[2].startswith('hip') or f[2].startswith('__hip'))})


def main():
    real, lib = sys.argv[1], sys.argv[2]
    dest = sys.argv[3] if len(sys.argv) > 3 else 'amd/src/libamdhip64.map'
    known = versions(real)
    nodes = {k: [] for k in ORDER}
    for name in ours(lib):
        nodes[known.get(name, 'hip_4.2')].append(name)
    text = [HEADER]
    prev = None
    for k in ORDER:
        body = ''.join(f'    {s};\n' for s in sorted(nodes[k]))
        glob = '  global:\n' + body if body else ''
        loc = '  local: *;\n' if k == 'hip_4.2' else ''
        text.append(f'{k} {{\n{glob}{loc}}}' + (f' {prev};' if prev else ';'))
        prev = k
    open(dest, 'w').write('\n'.join(text) + '\n')


if __name__ == '__main__':
    main()
