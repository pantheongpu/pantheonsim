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
  // Every fetch comes through here; once decoded, an acquire load is all it
  // costs (std::call_once's own fast path showed in profiles).
  if (decoded_->ready.load(std::memory_order_acquire)) [[likely]] return decoded_->instrs[i];
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
    decoded_->ready.store(true, std::memory_order_release);
  });
  return decoded_->instrs[i];
}

namespace {

// The functions a module may call that VirtualGPU provides itself.
const char* const kBuiltins[] = {"vprintf", "malloc", "free", "__assertfail", "cudaGraphSetConditional",
                                 "cudaGraphLaunch"};

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

// Writes `value` into the instruction a code relocation names: the low (56)
// or high (57) half of an address in a MOV's 32-bit immediate, a
// CALL.ABS.NOINC's target, 49 bits at 32 (58), or from sm_90 in 4-byte units
// at 16-23 and 34-80 (75).
static void patch_code(CubinSection& s, const CubinReloc& r, uint64_t value) {
  uint64_t hi, lo;
  std::memcpy(&lo, &s.bytes[r.offset], 8);
  std::memcpy(&hi, &s.bytes[r.offset + 8], 8);
  if (r.type == 58) {          // bits 32-80
    lo = (lo & 0xffffffffull) | (value << 32);
    hi = (hi & ~uint64_t{0x1ffff}) | ((value >> 32) & 0x1ffff);
  } else if (r.type == 75) {   // words: the low eight bits at 16-23, the rest at 34-80
    const uint64_t words = value >> 2;
    lo = (lo & ~(uint64_t{0xff} << 16) & 0x3ffffffffull) | ((words & 0xff) << 16) | ((words >> 8) << 34);
    hi = (hi & ~uint64_t{0x1ffff}) | ((words >> 8 >> 30) & 0x1ffff);
  } else {                     // bits 32-63
    const uint64_t half = r.type == 56 ? (value & 0xffffffffull) : (value >> 32);
    lo = (lo & 0xffffffffull) | (half << 32);
  }
  std::memcpy(&s.bytes[r.offset], &lo, 8);
  std::memcpy(&s.bytes[r.offset + 8], &hi, 8);
}

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
      if (sym.managed) m->managed.push_back(sym.name);
    }
    for (size_t i = 0; i < sizeof kBuiltins / sizeof *kBuiltins; ++i)
      m->builtins[kBuiltinBase + 16 * i] = kBuiltins[i];

    // A relocation's symbol: a variable, a function VirtualGPU provides, a
    // function's code, or a section. *in_code is set for code, whose offset
    // in its section is *code_off.
    const auto resolve = [&](const std::string& name, bool* in_code, uint64_t* code_off) -> uint64_t {
      *in_code = false;
      if (const auto v = m->symbol_va.find(name); v != m->symbol_va.end()) return v->second;
      for (const auto& [addr, b] : m->builtins)
        if (b == name) return addr;
      for (const CubinSymbol& sym : c.symbols) {
        if (sym.name != name || sym.section.empty()) continue;
        if (const auto code = m->code_index.find(sym.section); code != m->code_index.end()) {
          *in_code = true;
          *code_off = sym.value;
          return m->code[code->second].base + sym.value;
        }
        if (const auto sec = section_va.find(sym.section); sec != section_va.end()) return sec->second + sym.value;
      }
      throw Error(Err::NotFound, "cubin: relocation against unknown symbol " + name);
    };

    // Relocations. In data, R_CUDA_64 (2): an address. In code, which CUDA
    // 12.0's ptxas writes for addresses CUDA 13's loads from bank 4: the low
    // (56) or high (57) half of one in a MOV's 32-bit immediate, and a
    // CALL.ABS.NOINC's target, 49 bits at 32 (58), or from sm_90 in 4-byte
    // units at 16-23 and 34-80 (75). Code addresses are absolute (see RET).
    // Any other kind is refused by name rather than left
    // unpatched. The debug sections a -G build carries (.debug_line,
    // .debug_frame, ...) are never loaded, so their relocations do not matter.
    for (CubinSection& s : m->cubin.sections) {
      if (s.relocs.empty() || s.name.rfind(".debug_", 0) == 0 || s.name.rfind(".nv_debug", 0) == 0) continue;
      const bool code = m->code_index.count(s.name) != 0;
      const auto dst = section_va.find(s.name);
      if (!code && dst == section_va.end())
        throw Error(Err::UnsupportedPtx, "cubin: relocations in " + s.name + " are not supported yet");
      for (const CubinReloc& r : s.relocs) {
        // In data, R_CUDA_G64 (4) is an address too: NVIDIA's device link
        // writes it for a __device__ pointer initialised to a variable's
        // address (&array[2]), which nvJitLink's output keeps for the loader.
        const bool known = code ? (r.type == 56 || r.type == 57 || r.type == 58 || r.type == 75)
                                : (r.type == 2 || r.type == 4);
        if (!known)
          throw Error(Err::UnsupportedPtx,
                      "cubin: relocation type " + std::to_string(r.type) + " in " + s.name + " is not supported yet");
        bool in_code = false;
        uint64_t code_off = 0;
        const uint64_t target = resolve(r.symbol, &in_code, &code_off);
        if (!code) {
          const uint64_t value = target + static_cast<uint64_t>(r.addend);
          mem.write(dst->second + r.offset, &value, 8);
          continue;
        }
        if (r.offset + 16 > s.bytes.size())
          throw Error(Err::InvalidValue, "cubin: a relocation past the end of " + s.name);
        // A code address is written absolute: a -G build returns across
        // sections (each function in its own) through MOV'd addresses of
        // the caller plus an offset, which RET then takes as they are.
        (void)code_off;
        patch_code(s, r, target + static_cast<uint64_t>(r.addend));
      }
    }
  } catch (...) {
    unload(*m, mem);
    throw;
  }
  m->section_va = std::move(section_va);
  return m;
}

void rebind(Module& m, MemoryManager& mem, const std::string& name, uint64_t va) {
  const auto it = m.symbol_va.find(name);
  if (it == m.symbol_va.end()) throw Error(Err::NotFound, "cubin: no variable " + name + " to move");
  it->second = va;
  for (CubinSection& s : m.cubin.sections) {
    if (s.name.rfind(".debug_", 0) == 0 || s.name.rfind(".nv_debug", 0) == 0) continue;
    const bool code = m.code_index.count(s.name) != 0;
    const auto dst = m.section_va.find(s.name);
    for (const CubinReloc& r : s.relocs) {
      if (r.symbol != name) continue;
      const uint64_t value = va + static_cast<uint64_t>(r.addend);
      if (code) patch_code(s, r, value);
      else if (dst != m.section_va.end()) mem.write(dst->second + r.offset, &value, 8);
    }
  }
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
  const auto add = [&](const std::string& callee) {
    if (callee == kernel || std::find(out.begin(), out.end(), callee) != out.end()) return;
    out.push_back(callee);
    todo.push_back(callee);
  };
  while (!todo.empty()) {
    const std::string f = todo.back();
    todo.pop_back();
    for (const auto& [caller, callee] : m.cubin.calls)
      if (caller == f) add(callee);
    // CUDA 12.0's cubins name a call to malloc, free or vprintf only in the
    // relocation of its CALL.ABS.NOINC, not in the call graph.
    for (const CubinSection& s : m.cubin.sections)
      if (s.name == ".text." + f)
        for (const CubinReloc& r : s.relocs)
          if (r.type == 58 || r.type == 75) add(r.symbol);
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
