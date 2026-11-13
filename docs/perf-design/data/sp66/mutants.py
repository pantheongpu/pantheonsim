#!/usr/bin/env python3
"""Round sp66: mutants of what the round added to the host-GPU mode's logic: the card image (include/vgpu/host_gpu_image.hpp), the code
relocation writer it shares with the loader (include/vgpu/sass/reloc_patch.hpp), and in include/vgpu/host_gpu.hpp the kinds of external
functions, what each mode admits of them, the texture instructions, the printf switch and the atomics default. Each is a copy of one header with
one change, put in a directory outside the tree that comes first on the include path; tests/unit/test_host_gpu_logic.cpp is compiled against it
and must fail.

  mutants.py --src <tree> --work <dir outside the tree> [--only N] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse
import fcntl
import os
import subprocess
import time

IM = "vgpu/host_gpu_image.hpp"
RP = "vgpu/sass/reloc_patch.hpp"
HG = "vgpu/host_gpu.hpp"
M = [
    # the card image
    (IM, "image: no relocation is applied", "      if (!base) {\n        if (kept", "      base = std::nullopt;\n      if (!base) {\n        if (kept"),
    (IM, "image: the symbol's offset left out", "const uint64_t value = *base + st_value + r_addend;", "const uint64_t value = *base + r_addend;"),
    (IM, "image: the addend left out", "const uint64_t value = *base + st_value + r_addend;", "const uint64_t value = *base + st_value;"),
    (IM, "image: REL entries read as RELA", "const uint64_t es = sh[i].type == kRela ? 24 : 16;", "const uint64_t es = 24;"),
    (IM, "image: the table keeps its size", "    std::memcpy(&out.bytes[shoff + i * shentsize + 32], &new_size, 8);\n", "    (void)new_size;\n"),
    (IM, "image: the applied entry is kept too", "      ++out.applied;\n    }", "      ++out.applied;\n      if (kept * es != o - sh[i].offset) std::memmove(&out.bytes[sh[i].offset + kept * es], &out.bytes[o], es);\n      ++kept;\n    }"),
    (IM, "image: entries left are not moved down", "        if (kept * es != o - sh[i].offset) std::memmove(&out.bytes[sh[i].offset + kept * es], &out.bytes[o], es);\n        ++kept;\n        continue;", "        ++kept;\n        continue;"),
    (IM, "image: data relocations not written", "        std::memcpy(&out.bytes[sh[dst].offset + r_offset], &value, 8);\n", "        (void)value;\n"),
    (IM, "image: code relocations not written", "        sass::patch_code_word(&out.bytes[sh[dst].offset + r_offset], static_cast<uint32_t>(type), value);\n", "        (void)value;\n"),
    (IM, "image: G64 not taken", "} else if (!code && (type == 2 || type == 4)) {", "} else if (!code && type == 2) {"),
    (IM, "image: any data type taken", "} else if (!code && (type == 2 || type == 4)) {", "} else if (!code) {"),
    (IM, "image: any code type taken", "if (code && sass::code_reloc_type(static_cast<uint32_t>(type))) {", "if (code) {"),
    (IM, "image: a code type in data taken", "if (code && sass::code_reloc_type(static_cast<uint32_t>(type))) {", "if (sass::code_reloc_type(static_cast<uint32_t>(type))) {"),
    (IM, "image: NOBITS destinations written", "      if (sh[dst].type == kNobits) return fail(", "      if (false) return fail("),
    (IM, "image: no bound on code", "if (r_offset > sh[dst].size || sh[dst].size - r_offset < 16) return fail(\"a relocation past the end of \" + dst_name);\n        sass::", "sass::"),
    (IM, "image: a code bound of 8", "sh[dst].size - r_offset < 16) return fail", "sh[dst].size - r_offset < 8) return fail"),
    (IM, "image: no bound on data", "if (r_offset > sh[dst].size || sh[dst].size - r_offset < 8) return fail(\"a relocation past the end of \" + dst_name);\n        std::memcpy", "std::memcpy"),
    (IM, "image: a data bound of 9", "sh[dst].size - r_offset < 8) return fail(\"a relocation past the end of \" + dst_name);\n        std::memcpy", "sh[dst].size - r_offset < 9) return fail(\"a relocation past the end of \" + dst_name);\n        std::memcpy"),
    (IM, "image: the destination is the table", "const bool code = dst_name.rfind(\".text.\", 0) == 0;", "const bool code = name_of(i).rfind(\".text.\", 0) == 0;"),
    (IM, "image: the symbol's section is the destination", "if (st_shndx != 0 && st_shndx < shnum) base = var_va(name_of(st_shndx));", "if (st_shndx != 0 && st_shndx < shnum) base = var_va(name_of(dst));"),
    (IM, "image: the class byte not checked", "|| !rd(4, 1, &cls) || cls != 2)", "|| !rd(4, 1, &cls))"),
    (IM, "image: the header table not bounded", "      shstrndx >= shnum || shoff > n || shnum * shentsize > n - shoff)", "      shstrndx >= shnum || shoff > n)"),
    (IM, "image: a failure keeps the bytes", "    out.bytes.clear();\n    return out;", "    return out;"),
    (IM, "image: the value is the base alone", "const uint64_t value = *base + st_value + r_addend;", "const uint64_t value = *base;"),
    # the code writer
    (RP, "patch: 56 writes the high half", "type == 56 || type == 62 ? (value & 0xffffffffull)", "type == 62 ? (value & 0xffffffffull)"),
    (RP, "patch: 62 writes the high half", "type == 56 || type == 62 ? (value & 0xffffffffull)", "type == 56 ? (value & 0xffffffffull)"),
    (RP, "patch: 58's high bits dropped", "    hi = (hi & ~uint64_t{0x1ffff}) | ((value >> 32) & 0x1ffff);\n  } else if (type == 75)", "  } else if (type == 75)"),
    (RP, "patch: 75 not in words", "    const uint64_t words = value >> 2;", "    const uint64_t words = value;"),
    (RP, "patch: 75's low byte at 8", "((words & 0xff) << 16) | ((words >> 8) << 34);", "((words & 0xff) << 8) | ((words >> 8) << 34);"),
    (RP, "patch: the low word not kept", "    lo = (lo & 0xffffffffull) | (half << 32);", "    lo = half << 32;"),
    (RP, "patch: 63 not a code type", "type == 62 || type == 63 || type == 75; }", "type == 62 || type == 75; }"),
    (RP, "patch: 35 a code type", "type == 62 || type == 63 || type == 75; }", "type == 62 || type == 63 || type == 75 || type == 35; }"),
    # external functions, textures, switches
    (HG, "extern: printf admitted in exact mode", "return k == ExternKind::Printf && mode == Mode::Turbo && !printf_off;", "return k == ExternKind::Printf && mode != Mode::Off && !printf_off;"),
    (HG, "extern: the printf switch ignored", "return k == ExternKind::Printf && mode == Mode::Turbo && !printf_off;", "return k == ExternKind::Printf && mode == Mode::Turbo;"),
    (HG, "extern: the heap admitted", "return k == ExternKind::Printf && mode == Mode::Turbo && !printf_off;", "return (k == ExternKind::Printf || k == ExternKind::Heap) && mode == Mode::Turbo && !printf_off;"),
    (HG, "extern: assert is unknown", "  if (name == \"__assertfail\" || name == \"__assert_fail\") return ExternKind::Assert;\n", ""),
    (HG, "extern: free is unknown", "if (name == \"malloc\" || name == \"free\") return ExternKind::Heap;", "if (name == \"malloc\") return ExternKind::Heap;"),
    (HG, "extern: the device runtime second", "  if (device_runtime) return ExternKind::DeviceLaunch;\n  if (name == \"vprintf\") return ExternKind::Printf;", "  if (name == \"vprintf\") return ExternKind::Printf;\n  if (device_runtime) return ExternKind::DeviceLaunch;"),
    (HG, "extern: printf by prefix", "if (name == \"vprintf\") return ExternKind::Printf;", "if (name.rfind(\"vprintf\", 0) == 0) return ExternKind::Printf;"),
    (HG, "texture: TLD not named", "\"TEX\", \"TEXS\", \"TLD\", \"TLDS\"", "\"TEX\", \"TEXS\", \"TLDS\""),
    (HG, "texture: SUST not named", "\"SULD\", \"SUST\", \"SUATOM\"", "\"SULD\", \"SUATOM\""),
    (HG, "texture: by prefix", "  for (const char* t : kTex)\n    if (m == t) return true;", "  for (const char* t : kTex)\n    if (m.rfind(t, 0) == 0) return true;"),
    (HG, "switch: printf is always a bad name", "    else if (name == \"printf\" && printf_off) *printf_off = true;\n", ""),
    (HG, "switch: printf accepted with nowhere to put it", "    else if (name == \"printf\" && printf_off) *printf_off = true;", "    else if (name == \"printf\") { if (printf_off) *printf_off = true; }"),
    (HG, "switch: printf not reset", "  if (printf_off) *printf_off = false;\n", ""),
    (HG, "entry: anywhere in the section", "count > 0 && static_cast<uint64_t>(target) == base;", "count > 0 && static_cast<uint64_t>(target) >= base;"),
    (HG, "entry: a relative call followed", "return !relative && to_label && target >= 0 && count > 0", "return to_label && target >= 0 && count > 0"),
    (HG, "entry: an empty section followed", "target >= 0 && count > 0 && static_cast<uint64_t>(target) == base;", "target >= 0 && static_cast<uint64_t>(target) == base;"),
    (HG, "switch: atomics off by default", "  bool atomics = true;\n  ClassMask off = 0;", "  bool atomics = false;\n  ClassMask off = 0;"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    ap.add_argument("--hold", type=float, default=200)
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    work = os.path.abspath(a.work)
    assert not work.startswith(src + os.sep), "the mutants go outside the tree"
    os.makedirs(os.path.join(work, "inc/vgpu/sass"), exist_ok=True)
    text = {h: open(os.path.join(src, "include", h)).read() for h in (IM, RP, HG)}
    pre = ["nice", "-n", "19"]
    fd = os.open(a.lock, os.O_RDWR | os.O_CREAT, 0o666) if a.lock else None
    since = None
    killed = 0
    todo = [(i, m) for i, m in enumerate(M) if a.only < 0 or i == a.only]
    for i, (h, name, old, new) in todo:
        if text[h].count(old) != 1:
            print("%2d NOT APPLIED (%d matches): %s" % (i, text[h].count(old), name), flush=True)
            continue
        if fd is not None and since is not None and time.monotonic() - since > a.hold:   # the machine's build lock: given up now and then
            fcntl.flock(fd, fcntl.LOCK_UN)
            since = None
            time.sleep(1)
        if fd is not None and since is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
            since = time.monotonic()
        for other in (IM, RP, HG):
            p = os.path.join(work, "inc", other)
            if os.path.exists(p):
                os.remove(p)
        open(os.path.join(work, "inc", h), "w").write(text[h].replace(old, new))
        exe = os.path.join(work, "mut%d" % i)
        c = subprocess.run(pre + ["g++", "-std=c++20", "-O1", "-I", os.path.join(work, "inc"), "-I", os.path.join(src, "include"), "-I", os.path.join(src, "tests/framework"),
                                  os.path.join(src, "tests/unit/test_host_gpu_logic.cpp"), "-o", exe], capture_output=True, text=True)
        if c.returncode != 0:
            print("%2d killed (does not compile): %s" % (i, name), flush=True)
            killed += 1
            continue
        r = subprocess.run([exe], capture_output=True, text=True)
        fails = [l.split("]", 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("[ FAIL ]")]
        os.remove(exe)
        if r.returncode != 0:
            killed += 1
            print("%2d killed by %s: %s" % (i, ", ".join(fails[:3]) or "exit %d" % r.returncode, name), flush=True)
        else:
            print("%2d SURVIVED: %s" % (i, name), flush=True)
    if fd is not None and since is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    print("%d of %d killed" % (killed, len(todo)))


if __name__ == "__main__":
    main()
