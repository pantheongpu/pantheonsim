// Loading a cubin: decoding its code, and giving its constant banks and
// variables device memory with relocations resolved. See vgpu/sass/exec.hpp.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/sass/exec.hpp"

namespace vgpu::sass {

bool runs_on(int cubin_sm, bool arch_specific, int device_sm) {
  if (arch_specific) return cubin_sm == device_sm;
  return cubin_sm / 10 == device_sm / 10 && cubin_sm <= device_sm;
}

const Code* Module::code_at(uint64_t addr) const {
  for (const Code& c : code)
    if (addr >= c.base && addr < c.base + 16 * c.count) return &c;
  return nullptr;
}

// An instruction the decoder does not know is kept as an unknown one and
// reported if it is ever reached -- a kernel may carry code no launch executes.
const Instr& Code::instr(size_t i) const {
  std::call_once(decoded_->once, [&] {
    std::vector<Instr>& out = decoded_->instrs;
    out.reserve(count);
    for (size_t k = 0; k < count; ++k) {
      Word w;
      std::memcpy(&w.lo, &(*bytes)[16 * k], 8);
      std::memcpy(&w.hi, &(*bytes)[16 * k + 8], 8);
      try {
        out.push_back(decode(w, 16 * k, sm));
      } catch (const Error& e) {
        Instr bad;
        bad.w = w;
        bad.pc = 16 * k;
        bad.sm = sm;
        bad.op = Op::Unknown;
        bad.mnemonic = e.what();
        out.push_back(std::move(bad));
      }
    }
  });
  return decoded_->instrs[i];
}

namespace {

// The functions a module may call that VirtualGPU provides itself.
const char* const kBuiltins[] = {"vprintf", "malloc", "free", "__assertfail"};

bool is_bank(const std::string& name, unsigned* bank) {
  // ".nv.constant<N>" or ".nv.constant<N>.<kernel>"
  if (name.rfind(".nv.constant", 0) != 0) return false;
  size_t i = 12;
  unsigned n = 0;
  bool digits = false;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    n = n * 10 + static_cast<unsigned>(name[i++] - '0');
    digits = true;
  }
  if (!digits || (i < name.size() && name[i] != '.')) return false;
  *bank = n;
  return true;
}

}  // namespace

std::shared_ptr<Module> load(const uint8_t* image, size_t size, MemoryManager& mem, const DeviceProfile& profile) {
  auto m = std::make_shared<Module>();
  m->cubin = parse_cubin(image, size);
  const Cubin& c = m->cubin;
  m->sm = c.sm;
  const int device_sm = profile.cc_major * 10 + profile.cc_minor;
  if (!runs_on(c.sm, c.arch_specific, device_sm))
    throw Error(Err::UnsupportedPtx, "cubin built for sm_" + std::to_string(c.sm) + " does not run on " + profile.id +
                                         " (sm_" + std::to_string(device_sm) + ")");

  // Code: every .text section, placed in the code window; each is decoded
  // when it first runs (Code::instr). The cubin's sections outlive them.
  uint64_t code_va = kCodeBase;
  for (const CubinSection& s : c.sections) {
    if (s.name.rfind(".text.", 0) != 0) continue;
    Code code;
    code.section = s.name;
    code.base = code_va;
    code.count = s.bytes.size() / 16;
    code.sm = c.sm;
    code.bytes = &s.bytes;
    const size_t n = code.count;
    code_va += (16 * n + 0xff) & ~uint64_t{0xff};
    m->code_index[s.name] = m->code.size();
    m->code.push_back(std::move(code));
  }
  for (size_t i = 0; i < c.kernels.size(); ++i) {
    if (!m->code_index.count(c.kernels[i].text_section))
      throw Error(Err::InvalidValue, "cubin: kernel " + c.kernels[i].name + " has no code");
    m->kernel_index[c.kernels[i].name] = i;
  }

  // Device memory for everything a kernel addresses: constant banks (except
  // bank 0, which each launch builds) and the module's variables.
  std::map<std::string, uint64_t> section_va;
  try {
    for (const CubinSection& s : c.sections) {
      unsigned bank = 0;
      const bool is_const = is_bank(s.name, &bank) && bank != 0;
      const bool is_var = s.name == ".nv.global" || s.name == ".nv.global.init";
      if (!is_const && !is_var) continue;
      if (s.size == 0) continue;
      const uint64_t va = mem.alloc(s.size);
      m->allocations.push_back(va);
      if (!s.bytes.empty()) mem.write(va, s.bytes.data(), s.bytes.size());
      section_va[s.name] = va;
      if (is_const) m->bank_va[s.name] = va;
    }
    for (const CubinSymbol& sym : c.symbols) {
      if (sym.function || sym.name.empty() || sym.section.empty()) continue;
      const auto it = section_va.find(sym.section);
      if (it == section_va.end() || sym.name[0] == '.') continue;
      m->symbol_va[sym.name] = it->second + sym.value;
      m->symbol_size[sym.name] = sym.size;
    }
    for (size_t i = 0; i < sizeof kBuiltins / sizeof *kBuiltins; ++i)
      m->builtins[kBuiltinBase + 16 * i] = kBuiltins[i];

    // Relocations into device memory. R_CUDA_64 is the only kind cubins built
    // by the toolkits checked here carry in data; one in code, or of another
    // kind, is refused by name rather than left unpatched.
    for (const CubinSection& s : c.sections) {
      if (s.relocs.empty() || s.name == ".debug_frame") continue;
      const auto dst = section_va.find(s.name);
      if (dst == section_va.end())
        throw Error(Err::UnsupportedPtx, "cubin: relocations in " + s.name + " are not supported yet");
      for (const CubinReloc& r : s.relocs) {
        if (r.type != 2)
          throw Error(Err::UnsupportedPtx,
                      "cubin: relocation type " + std::to_string(r.type) + " in " + s.name + " is not supported yet");
        uint64_t target = 0;
        if (const auto v = m->symbol_va.find(r.symbol); v != m->symbol_va.end()) {
          target = v->second;
        } else {
          bool found = false;
          for (const auto& [addr, name] : m->builtins)
            if (name == r.symbol) {
              target = addr;
              found = true;
            }
          for (const CubinSymbol& sym : c.symbols)
            if (!found && sym.name == r.symbol && !sym.section.empty()) {
              if (const auto code = m->code_index.find(sym.section); code != m->code_index.end()) {
                target = m->code[code->second].base + sym.value;
                found = true;
              } else if (const auto sec = section_va.find(sym.section); sec != section_va.end()) {
                target = sec->second + sym.value;
                found = true;
              }
            }
          if (!found) throw Error(Err::NotFound, "cubin: relocation against unknown symbol " + r.symbol);
        }
        const uint64_t value = target + static_cast<uint64_t>(r.addend);
        mem.write(dst->second + r.offset, &value, 8);
      }
    }
  } catch (...) {
    unload(*m, mem);
    throw;
  }
  return m;
}

bool runs_instr(const Instr& ins);   // exec.cpp

std::string unsupported(const uint8_t* image, size_t size) {
  const Cubin c = parse_cubin(image, size);
  // VGPU_SASS_REFUSE=<op>: treat that op as unsupported, for testing the
  // fallback to PTX.
  const char* refuse = std::getenv("VGPU_SASS_REFUSE");
  for (const CubinSection& s : c.sections) {
    if (s.name.rfind(".text.", 0) != 0) continue;
    for (size_t i = 0; i + 16 <= s.bytes.size(); i += 16) {
      Word w;
      std::memcpy(&w.lo, &s.bytes[i], 8);
      std::memcpy(&w.hi, &s.bytes[i + 8], 8);
      char at[96];
      std::snprintf(at, sizeof at, "sm_%d %s+0x%zx: ", c.sm, s.name.c_str() + 6, i);
      try {
        const Instr ins = decode(w, i, c.sm);
        if (!runs_instr(ins) || (refuse && *refuse && ins.mnemonic == refuse)) return at + to_text(ins);
      } catch (const Error& e) {
        return at + e.message();
      }
    }
  }
  return {};
}

std::vector<std::string> reachable(const Module& m, const std::string& kernel) {
  std::vector<std::string> out, todo{kernel};
  while (!todo.empty()) {
    const std::string f = todo.back();
    todo.pop_back();
    for (const auto& [caller, callee] : m.cubin.calls)
      if (caller == f && std::find(out.begin(), out.end(), callee) == out.end()) {
        out.push_back(callee);
        todo.push_back(callee);
      }
  }
  return out;
}

void unload(Module& m, MemoryManager& mem) {
  for (uint64_t va : m.allocations) {
    try {
      mem.free(va);
    } catch (const Error&) {
      // already gone (a device reset)
    }
  }
  m.allocations.clear();
}

}  // namespace vgpu::sass
