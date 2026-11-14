#!/usr/bin/env python3
"""Round sp71: mutants of the hiprtc cache (amd/src/hiprtc_cache.cpp). Each is one edit of a copy of the file, compiled into a
libhiprtc.so.6 of its own in a directory outside the tree; the build's test_hiprtc_cache (the stand-in for ROCm's library, no ROCm
needed) is run with that directory first on the library path and must fail. The library is one file, so a batch of mutants is
compiled under one hold of the machine's build lock and the tests run outside it.

  mutants.py --src <tree> --build <tree>/build --work <dir outside the tree> [--only N,N] [--batch 8] [--lock /tmp/pantheonsim-builder.lock]
"""
import argparse
import os
import subprocess

F = "amd/src/hiprtc_cache.cpp"
M = [
    # the key
    ("key: without the source", "  put(&k, p.src);\n", ""),
    ("key: without the program's name", "  put(&k, p.name);\n", ""),
    ("key: without the options", "  put(&k, options);\n", ""),
    ("key: without the headers' text", "  put(&k, p.headers);\n", ""),
    ("key: without the headers' names", "  put(&k, p.include_names);\n", ""),
    ("key: without the name expressions", "  put(&k, p.name_exprs);\n", ""),
    ("key: without the device's architecture", "  put(&k, arch);\n", ""),
    ("key: without the environment", "  put(&k, compile_environment());\n", ""),
    ("key: without the toolchain", "  put(&k, toolchain_text);\n", ""),
    ("key: without the hip_runtime.h files", "  put(&k, extra);\n", ""),
    ("toolchain: without the device bitcode and configuration files", "  for (const std::string& l : lines) text += l + \"\\n\";\n", ""),
    ("toolchain: without the compiler library", "    text += \"comgr \" + stat_line(t.comgr) + \" \" + build_id_of(head) + \"\\n\";\n", ""),
    ("toolchain: the built-in header library by name only",
     "    text += \"builtins \" + t.builtins + \" \" + build_id_of(t.header_bytes) + \" \" + hex16(fnv(t.header_bytes.data(), t.header_bytes.size())) +\n"
     "            hex16(mix(t.header_bytes.data(), t.header_bytes.size())) + \"\\n\";\n",
     "    text += \"builtins \" + t.builtins + \"\\n\";\n"),
    ("environment: only the HIPRTC_ variables", "    for (const char* p : kPrefixes) in = in || name.rfind(p, 0) == 0;", "    in = in || name.rfind(\"HIPRTC_\", 0) == 0;"),
    # an entry's file
    ("entry: used on its name alone (the key not compared)", "  if (h.key_size != key.size() || std::memcmp(k, key.data(), key.size()) != 0) {", "  if (false) {"),
    ("entry: the body's check not made", " || fnv(b, h.body_size) != h.body_check) return bad();", ") return bad();"),
    ("entry: its length not held to its head", "      sizeof h + h.key_size + h.body_size != all.size())\n    return bad();", "      false)\n    return bad();"),
    ("entry: any format", " || h.format != kFormat", ""),
    ("entry: any magic", "std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || ", ""),
    ("entry: a file others may write is read", "  if (::fstat(fd, &st) != 0 || !ours_alone(st, S_IFREG)) return bad();", "  if (::fstat(fd, &st) != 0) return bad();"),
    ("directory: one others may write is used", "    return ours_alone(st, S_IFDIR) ? d : std::string();", "    return d;"),
    ("cap: never trimmed", "  if (!cap || total <= cap) return;", "  if (true) return;"),
    ("cap: trimmed to nothing", "    if (total <= cap / 10 * 9) break;", "    if (total == 0) break;"),
    ("cap: 0 is a cap of nothing", "  if (!cap || total <= cap) return;", "  if (total <= cap) return;"),
    ("mode: mem writes files", "  const bool files = mode() == Mode::Disk && t->known && !cache_dir().empty();", "  const bool files = t->known && !cache_dir().empty();"),
    ("mode: 0 still answers from the cache", "  bool plain = mode() != Mode::Off && prog && src && name", "  bool plain = prog && src && name"),
    # what is cacheable
    ("cacheable: a failed compile is kept", "  if (c != kSuccess || p->state != Prog::State::Real) return c;", "  if (p->state != Prog::State::Real) return c;"),
    ("cacheable: any option", "    if (!pure_option(o, &include_path)) return \"an option outside the list (\" + o.substr(0, 24) + \")\";", "    (void)pure_option(o, &include_path);"),
    ("cacheable: a reserved macro", "    if (name.empty() || name[0] == '_' || name.rfind(\"HIP\", 0) == 0 || name.rfind(\"CUDA\", 0) == 0) return false;", "    if (name.empty()) return false;"),
    ("cacheable: options handed to another tool", "  if (o.rfind(\"-Wl,\", 0) == 0 || o.rfind(\"-Wa,\", 0) == 0 || o.rfind(\"-Wp,\", 0) == 0) return false;   // (handed to another tool)\n", ""),
    ("cacheable: an include that was not given", "      else external = true;", "      else external = false;"),
    ("cacheable: an include that is not a file name", "      if (e >= text.size() || (text[e] != '<' && text[e] != '\"')) return \"an #include that is not a file name\";",
     "      if (e >= text.size() || (text[e] != '<' && text[e] != '\"')) continue;"),
    ("cacheable: continued lines not joined", "    if (s[i] == '\\\\' && i + 1 < s.size() && s[i + 1] == '\\n') { ++i; continue; }\n", ""),
    ("cacheable: __DATE__ and __FILE__", "    for (const char* word : {\"__DATE__\", \"__TIME__\", \"__TIMESTAMP__\", \"__FILE__\", \"__BASE_FILE__\", \"__has_include\", \"__has_embed\", \"?\" \"?=\", \"%:\"})",
     "    for (const char* word : {\"__TIME__\", \"__TIMESTAMP__\", \"__BASE_FILE__\", \"__has_include\", \"__has_embed\", \"?\" \"?=\", \"%:\"})"),
    ("cacheable: comments are read as code", "    if (c == '/' && i + 1 < joined.size() && joined[i + 1] == '/') {", "    if (false) {"),
    ("runtime header: with an include path", "    if (include_path || include_environment_set()) return \"<hip/hip_runtime.h> with an include path\";\n", ""),
    ("runtime header: with CPATH", "    if (include_path || include_environment_set()) return", "    if (include_path) return"),
    ("runtime header: any file", "    if (!read_file(p, &bytes, 1 << 22) || !wholly_guarded(bytes)) return false;", "    if (!read_file(p, &bytes, 1 << 22)) return false;"),
    ("runtime header: something after the guard", "      if (closed) return false;   // something after the guard's #endif\n", ""),
    ("runtime header: something before the guard", "        if (!opened) return false;   // something before the guard\n", ""),
    ("runtime header: any guard", "          if (word != \"ifndef\" || text.substr(a, b - a) != kGuard) return false;", "          if (word != \"ifndef\") return false;"),
    ("runtime header: nowhere is fine", "  return found > 0;", "  return true;"),
    ("runtime header: the built-in header's guard not looked for", "  if (t.header_bytes.find(std::string(\"#define \") + kGuard) == std::string::npos) return false;\n", ""),
    ("cacheable: hiprtc_runtime.h in the working directory", "  if (::stat(\"hiprtc_runtime.h\", &st) == 0) return \"hiprtc_runtime.h in the working directory\";\n", ""),
    ("cacheable: no device to ask", "  std::string why = arch.empty() ? std::string(\"no HIP runtime to ask for the device's architecture\") : not_cacheable(*p, opts, *t, &extra);",
     "  std::string why = not_cacheable(*p, opts, *t, &extra);"),
    # the answers
    ("answers: a name expression after a cached compile is taken", "  if (p && p->state == Prog::State::Fresh && name_expression) return p->name_exprs.push_back(name_expression), kSuccess;",
     "  if (p && p->state != Prog::State::Real && name_expression) return p->name_exprs.push_back(name_expression), kSuccess;"),
    ("answers: the lowered names in another order", "    e->lowered.push_back(lowered);", "    e->lowered.insert(e->lowered.begin(), lowered);"),
    ("answers: an unknown name gets the first one's", "        return false;   // (a name that was not added: the real library's answer)",
     "        return !p.entry->lowered.empty() && (*lowered_name = p.entry->lowered[0].c_str(), true);"),
    ("answers: the log's size is the log's length", "      prog, size != nullptr, [&](const Prog& p) { return *size = static_cast<size_t>(p.entry->log_size), true; },",
     "      prog, size != nullptr, [&](const Prog& p) { return *size = p.entry->log.empty() ? 0 : p.entry->log.size() - 1, true; },"),
    ("answers: one byte of code short", "[&](const Prog& p) { return std::memcpy(code, p.entry->code.data(), p.entry->code.size()), true; }",
     "[&](const Prog& p) { return std::memcpy(code, p.entry->code.data(), p.entry->code.size() - 1), true; }"),
    ("answers: a destroy clears the caller's handle (ROCm's library leaves it)", "  delete p;   // (the caller's handle is left as it was: ROCm's library does not clear it)\n  return c;",
     "  delete p;\n  *prog = nullptr;\n  return c;"),
    ("search: the simulator's own library is taken for the real one", "      if (::stat(c.c_str(), &st) != 0 || (!self.empty() && same_file(c, self))) continue;", "      if (::stat(c.c_str(), &st) != 0) continue;"),
    ("statistics: printed unasked", "    if (!(e && *e && *e != '0')) return false;", "    (void)e;"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lock", default="/tmp/pantheonsim-builder.lock")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",") if x}
    work = os.path.abspath(a.work)
    assert not work.startswith(os.path.abspath(a.src) + os.sep), "the mutants go outside the tree"
    text = open(os.path.join(a.src, F)).read()
    todo = [(i, m) for i, m in enumerate(M) if not only or i in only]
    killed = 0
    # The control: the file as it is, built the same way, must pass.
    d = os.path.join(work, "control")
    os.makedirs(d, exist_ok=True)
    subprocess.run(["flock", a.lock, "nice", "-n", "19", "sh", "-c",
                    "g++ -std=c++20 -O1 -fPIC -shared -fvisibility=hidden -o %s/libhiprtc.so.6 %s -ldl" % (d, os.path.join(a.src, F))], check=True)
    t = subprocess.run([os.path.join(a.build, "test_hiprtc_cache"), os.path.join(a.build, "hiprtc-fakes")], capture_output=True, text=True,
                       env=dict(os.environ, LD_LIBRARY_PATH=d))
    print("control: the unchanged file built this way %s" % ("passes" if t.returncode == 0 else "FAILS: " + t.stdout[-300:]), flush=True)
    if t.returncode != 0:
        return
    for at in range(0, len(todo), a.batch):
        batch = todo[at:at + a.batch]
        script = []
        for i, (name, old, new) in batch:
            d = os.path.join(work, "m%d" % i)
            os.makedirs(d, exist_ok=True)
            if text.count(old) != 1:
                continue
            open(os.path.join(d, "hiprtc_cache.cpp"), "w").write(text.replace(old, new))
            script.append("g++ -std=c++20 -O1 -fPIC -shared -fvisibility=hidden -o %s/libhiprtc.so.6 %s/hiprtc_cache.cpp -ldl 2> %s/build.log || rm -f %s/libhiprtc.so.6"
                          % (d, d, d, d))
        subprocess.run(["flock", a.lock, "nice", "-n", "19", "sh", "-c", "\n".join(script)], check=False)
        for i, (name, old, new) in batch:
            d = os.path.join(work, "m%d" % i)
            if text.count(old) != 1:
                print("%2d NOT APPLIED (%d matches): %s" % (i, text.count(old), name), flush=True)
                continue
            if not os.path.exists(os.path.join(d, "libhiprtc.so.6")):
                print("%2d DID NOT BUILD: %s" % (i, name), flush=True)
                continue
            env = dict(os.environ, LD_LIBRARY_PATH=d)
            try:
                t = subprocess.run([os.path.join(a.build, "test_hiprtc_cache"), os.path.join(a.build, "hiprtc-fakes")], capture_output=True, text=True,
                                   timeout=900, env=env)
                failed = [l[6:].strip() for l in t.stdout.splitlines() if l.startswith("FAIL ")]
                if t.returncode != 0 and failed:
                    killed += 1
                    print("%2d killed by %d check(s), first: \"%s\": %s" % (i, len(failed), failed[0][:90], name), flush=True)
                else:
                    print("%2d SURVIVED: %s" % (i, name), flush=True)
            except subprocess.TimeoutExpired:
                killed += 1
                print("%2d killed (the test did not end): %s" % (i, name), flush=True)
    print("%d of %d killed" % (killed, len(todo)), flush=True)


if __name__ == "__main__":
    main()
