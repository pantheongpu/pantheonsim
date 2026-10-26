// The PTX engine's tile route for wmma.mma on the shapes and types its tile (m16n16k16, f16 and bf16) does not take -- m32n8k16 and m8n32k16 of
// f16, bf16, s8 and u8, m16n16k8 of tf32, m8n8k32 of s4 and u4, m8n8k4 of f64 in all four rounding modes, f16 and f32 accumulators and results
// in every pairing -- against the general form (VGPU_FASTPATH=0), bit for bit, on an RTX A10 (sm_86: the card-measured chunked sum) and an A100
// (the plain f32 sum), with fragments of random bits drawn from the classes that break arithmetic (zeros of both signs, subnormals, infinities,
// NaNs, the extremes, wide and narrow exponents, integers at their limits, accumulators near the s32 limits). Single-bit and 16x16x16 forms are
// run too: the route leaves them to the general form, and must give the same answer by leaving. And mma.sync.m8n8k4 (Volta's f16 form: four 8x8x4
// products, f16 or f32 C and D in every pairing, all four layouts), whose products go through the same kernel.
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/ast.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(&b[0], &v, 8);
  return b;
}

uint32_t f32_class(std::mt19937& g, bool wide) {
  const uint32_t h = g();
  switch (h % 16) {
    case 0: return 0;
    case 1: return 0x80000000u;
    case 2: return h & 0x807fffffu;
    case 3: return 0x7f800000u | (h & 0x80000000u);
    case 4: return 0x7fc00000u | (h & 0x803fffffu);
    case 5: return 0x7f800001u | (h & 0x80000000u);
    case 6: return (h & 0x80000000u) | 0x7f7fffffu;
    default: {
      const uint32_t e = wide ? 1 + (h >> 8) % 254 : 118 + (h >> 8) % 20;
      return (h & 0x80000000u) | (e << 23) | (g() & 0x7fffffu);
    }
  }
}
uint16_t half_class(std::mt19937& g, bool wide) {
  const uint32_t h = g();
  switch (h % 14) {
    case 0: return 0;
    case 1: return 0x8000;
    case 2: return static_cast<uint16_t>((h & 0x8000u) | ((h >> 8) & 0x3ffu));
    case 3: return static_cast<uint16_t>(0x7c00u | (h & 0x8000u));
    case 4: return static_cast<uint16_t>(0x7e00u | (h & 0x83ffu));
    case 5: return static_cast<uint16_t>((h & 0x8000u) | 0x7bffu);
    default: {
      const uint32_t e = wide ? 1 + (h >> 8) % 30 : 10 + (h >> 8) % 10;
      return static_cast<uint16_t>((h & 0x8000u) | (e << 10) | (g() & 0x3ffu));
    }
  }
}
uint64_t f64_class(std::mt19937& g, bool wide, bool nan_ok) {
  const uint64_t h = (uint64_t{g()} << 32) | g();
  switch (h % 12) {
    case 0: return 0;
    case 1: return 0x8000000000000000ull;
    case 2: return h & 0x800fffffffffffffull;
    case 3: return nan_ok ? (0x7ff0000000000000ull | (h & 0x8000000000000000ull)) : 0x3ff0000000000000ull;
    case 4: return nan_ok ? (0x7ff8000000000000ull | (h >> 12)) : 0x4000000000000000ull;
    case 5: return (h & 0x8000000000000000ull) | 0x7fefffffffffffffull;
    default: {
      const uint64_t e = wide ? 1 + (h >> 20) % 2046 : 1013 + (h >> 20) % 20;
      return (h & 0x8000000000000000ull) | (e << 52) | (h & 0xfffffffffffffull);
    }
  }
}

enum class Kind { F16, BF16, TF32, Int, Bits, F64 };

struct Form {
  const char* name;
  const char* gpu;
  const char* shape;    // ".m32n8k16"
  const char* types;    // ".f32.f32" or ".f32.bf16.bf16.f32" (after the shape and the rounding)
  const char* suffix;   // ".satfinite", ".rn", ".xor.popc"
  Kind kind;
  bool c_f16 = false;   // C draws half-pairs
  bool wide64 = false;  // 64-bit registers
  bool mma = false;     // mma.sync.aligned.m8n8k4 (Volta's form) rather than wmma.mma
  uint32_t ra = 0, rb = 0, rc = 0, rd = 0;   // register counts, when the form says them (the mma form)
};

// (the register counts come from wmma_geom, as the parser takes them)
ptx::WmmaType type_of(const std::string& t) {
  using T = ptx::WmmaType;
  if (t == "f16") return T::F16;
  if (t == "bf16") return T::BF16;
  if (t == "tf32") return T::TF32;
  if (t == "f32") return T::F32;
  if (t == "f64") return T::F64;
  if (t == "s8") return T::S8;
  if (t == "u8") return T::U8;
  if (t == "s4") return T::S4;
  if (t == "u4") return T::U4;
  if (t == "b1") return T::B1;
  return T::S32;
}

const Form kForms[] = {
    {"f16 m32n8k16 f32/f32 a10", "nvidia/a10", ".m32n8k16", ".f32.f32", "", Kind::F16},
    {"f16 m32n8k16 f32/f32 a100", "nvidia/a100", ".m32n8k16", ".f32.f32", "", Kind::F16},
    {"f16 m8n32k16 f32/f32 a10", "nvidia/a10", ".m8n32k16", ".f32.f32", "", Kind::F16},
    {"f16 m8n32k16 f32/f32 a100", "nvidia/a100", ".m8n32k16", ".f32.f32", "", Kind::F16},
    {"f16 m32n8k16 f16/f16 a10", "nvidia/a10", ".m32n8k16", ".f16.f16", "", Kind::F16, true},
    {"f16 m8n32k16 f16/f16 a100", "nvidia/a100", ".m8n32k16", ".f16.f16", "", Kind::F16, true},
    {"f16 m32n8k16 f16/f32 a10", "nvidia/a10", ".m32n8k16", ".f16.f32", "", Kind::F16},
    {"f16 m32n8k16 f32/f16 a100", "nvidia/a100", ".m32n8k16", ".f32.f16", "", Kind::F16, true},
    {"f16 m16n16k16 f16/f16 a10", "nvidia/a10", ".m16n16k16", ".f16.f16", "", Kind::F16, true},
    {"f16 m16n16k16 f16/f32 a100", "nvidia/a100", ".m16n16k16", ".f16.f32", "", Kind::F16},
    {"bf16 m32n8k16 a10", "nvidia/a10", ".m32n8k16", ".f32.bf16.bf16.f32", "", Kind::BF16},
    {"bf16 m8n32k16 a100", "nvidia/a100", ".m8n32k16", ".f32.bf16.bf16.f32", "", Kind::BF16},
    {"bf16 m16n16k16 a10", "nvidia/a10", ".m16n16k16", ".f32.bf16.bf16.f32", "", Kind::BF16},
    {"tf32 m16n16k8 a10", "nvidia/a10", ".m16n16k8", ".f32.tf32.tf32.f32", "", Kind::TF32},
    {"tf32 m16n16k8 a100", "nvidia/a100", ".m16n16k8", ".f32.tf32.tf32.f32", "", Kind::TF32},
    {"s8 m16n16k16", "nvidia/a100", ".m16n16k16", ".s32.s8.s8.s32", "", Kind::Int},
    {"s8 m32n8k16 sat", "nvidia/a100", ".m32n8k16", ".s32.s8.s8.s32", ".satfinite", Kind::Int},
    {"u8 m8n32k16", "nvidia/a100", ".m8n32k16", ".s32.u8.u8.s32", "", Kind::Int},
    {"u8 m16n16k16 sat", "nvidia/a10", ".m16n16k16", ".s32.u8.u8.s32", ".satfinite", Kind::Int},
    {"s4 m8n8k32", "nvidia/a100", ".m8n8k32", ".s32.s4.s4.s32", "", Kind::Int},
    {"u4 m8n8k32 sat", "nvidia/a100", ".m8n8k32", ".s32.u4.u4.s32", ".satfinite", Kind::Int},
    {"b1 m8n8k128", "nvidia/a100", ".m8n8k128", ".s32.b1.b1.s32", ".xor.popc", Kind::Bits},
    {"f64 m8n8k4 rn", "nvidia/a100", ".m8n8k4", ".f64.f64.f64.f64", ".rn", Kind::F64, false, true},
    {"f64 m8n8k4 rz", "nvidia/a100", ".m8n8k4", ".f64.f64.f64.f64", ".rz", Kind::F64, false, true},
    {"f64 m8n8k4 rm", "nvidia/a100", ".m8n8k4", ".f64.f64.f64.f64", ".rm", Kind::F64, false, true},
    {"f64 m8n8k4 rp", "nvidia/a100", ".m8n8k4", ".f64.f64.f64.f64", ".rp", Kind::F64, false, true},
    {"mma m8n8k4 f32/f32", "nvidia/a10", ".m8n8k4", ".f32.f16.f16.f32", "", Kind::F16, false, false, true, 2, 2, 8, 8},
    {"mma m8n8k4 f16/f16", "nvidia/a10", ".m8n8k4", ".f16.f16.f16.f16", "", Kind::F16, true, false, true, 2, 2, 4, 4},
    {"mma m8n8k4 f32/f16", "nvidia/a100", ".m8n8k4", ".f32.f16.f16.f16", "", Kind::F16, true, false, true, 2, 2, 4, 8},
    {"mma m8n8k4 f16/f32", "nvidia/a100", ".m8n8k4", ".f16.f16.f16.f32", "", Kind::F16, false, false, true, 2, 2, 8, 4},
};

ptx::WmmaShape shape_of(const std::string& s) {
  unsigned m, n, k;
  std::sscanf(s.c_str(), ".m%un%uk%u", &m, &n, &k);
  ptx::WmmaShape sh;
  sh.m = m, sh.n = n, sh.k = k;
  return sh;
}

std::string regs(const char* p, uint32_t n) {
  std::string s = "{";
  for (uint32_t i = 0; i < n; ++i) s += std::string(i ? "," : "") + "%" + p + std::to_string(i);
  return s + "}";
}

// One launch: one warp (all lanes), the fragments loaded from `in` (A, B, C registers a lane), D stored after; D's words (`out`).
std::vector<uint64_t> run(const Form& f, const std::string& layouts, const std::vector<uint64_t>& in, uint32_t ra, uint32_t rb, uint32_t rc, uint32_t rd, uint32_t active) {
  const char* rt = f.wide64 ? "b64" : "b32";
  const uint32_t w = f.wide64 ? 8 : 4;
  std::string ptx = std::string(".version 8.3\n.target ") + (std::string(f.gpu) == "nvidia/a10" ? "sm_86" : "sm_80") + R"(
.address_size 64
.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .reg .pred %q;
    .reg .b32 %r<8>;
    .reg .)" + rt + R"( %a<40>, %b<40>, %c<40>, %d<40>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pin];
    ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %laneid;
    setp.lt.u32 %q, %r1, )" + std::to_string(active) + R"(;
    mul.wide.u32 %rd3, %r1, 1024;
    add.u64 %rd4, %rd1, %rd3;
    mul.wide.u32 %rd5, %r1, 512;
    add.u64 %rd6, %rd2, %rd5;
)";
  uint32_t off = 0;
  for (uint32_t i = 0; i < ra; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %a" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  for (uint32_t i = 0; i < rb; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %b" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  for (uint32_t i = 0; i < rc; ++i, off += w) ptx += std::string("    ld.global.") + rt + " %c" + std::to_string(i) + ", [%rd4+" + std::to_string(off) + "];\n";
  for (uint32_t i = 0; i < rd; ++i) ptx += std::string("    mov.") + rt + " %d" + std::to_string(i) + ", " + (f.wide64 ? "0xA5A5A5A5A5A5A5A5" : "0xA5A5A5A5") + ";\n";
  // (.xor.popc / .and.popc follow wmma.mma, a rounding mode follows the shape, .satfinite ends the types)
  const std::string suffix = f.suffix;
  const bool popc = suffix.find("popc") != std::string::npos, sat = suffix == ".satfinite";
  std::string lay_dotless = layouts;
  lay_dotless.pop_back();
  ptx += (f.mma ? std::string("    @%q mma.sync.aligned") + f.shape + "." + lay_dotless + f.types
                : std::string("    @%q wmma.mma") + (popc ? suffix : "") + ".sync.aligned." + layouts + f.shape + (!popc && !sat ? suffix : "") + f.types + (sat ? suffix : "")) +
         " " + regs("d", rd) + ", " + regs("a", ra) + ", " + regs("b", rb) + ", " + regs("c", rc) + ";\n";
  for (uint32_t i = 0; i < rd; ++i) ptx += std::string("    st.global.") + rt + " [%rd6+" + std::to_string(i * w) + "], %d" + std::to_string(i) + ";\n";
  ptx += "    ret;\n}\n";
  MemoryManager mem{1 << 22};
  DeviceProfile prof = load_gpu(f.gpu);
  auto m = ptx::parse(ptx);
  const uint64_t pin = mem.alloc(32 * 1024), pout = mem.alloc(32 * 512);
  std::vector<uint8_t> img(32 * 1024, 0);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    for (uint32_t i = 0; i < ra + rb + rc; ++i) {
      const uint64_t v = in[lane * 128 + i];
      std::memcpy(&img[lane * 1024 + i * w], &v, w);
    }
  }
  mem.write(pin, img.data(), img.size());
  std::vector<uint8_t> init(32 * 512, 0x5A);
  mem.write(pout, init.data(), init.size());
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(pin), arg_u64(pout)}, mem, prof);
  std::vector<uint64_t> out(32 * rd);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    std::vector<uint8_t> row(512);
    mem.read(pout + lane * 512, row.data(), row.size());
    for (uint32_t i = 0; i < rd; ++i) std::memcpy(&out[lane * rd + i], &row[i * w], w);
  }
  return out;
}

}  // namespace

VTEST(wmma_generic_tile_route_matches_the_general_form) {
  std::mt19937 g(0x3a17);
  long compared = 0;
  for (const Form& f : kForms) {
    // The register counts: A, B, C and D from the type list ("f32.f32": D, C; "s32.s8.s8.s32": D, A, B, C).
    std::vector<std::string> types;
    {
      std::string t = f.types;
      for (size_t at = 1; at < t.size();) {
        const size_t dot = t.find('.', at);
        types.push_back(t.substr(at, dot == std::string::npos ? std::string::npos : dot - at));
        at = dot == std::string::npos ? t.size() : dot + 1;
      }
    }
    const bool short_form = types.size() == 2 || f.mma;
    const ptx::WmmaType dt = type_of(types[0]), ct = type_of(types.back()), at = short_form ? ptx::WmmaType::F16 : type_of(types[1]);
    const ptx::WmmaShape sh = shape_of(f.shape);
    const uint32_t ra = f.mma ? f.ra : ptx::wmma_geom('a', sh, at).regs, rb = f.mma ? f.rb : ptx::wmma_geom('b', sh, at).regs, rc = f.mma ? f.rc : ptx::wmma_geom('c', sh, ct).regs,
                   rd = f.mma ? f.rd : ptx::wmma_geom('d', sh, dt).regs;
    for (const char* layouts : {"row.col.", "col.row.", "row.row.", "col.col."}) {
      if ((f.kind == Kind::Int || f.kind == Kind::Bits) && (at == ptx::WmmaType::S4 || at == ptx::WmmaType::U4 || at == ptx::WmmaType::B1) && std::string(layouts) != "row.col.") continue;
      for (int trial = 0; trial < 8; ++trial) {
        const bool wide = trial % 2 == 1;
        std::vector<uint64_t> in(32 * 128);
        for (uint32_t lane = 0; lane < 32; ++lane) {
          uint64_t* row = &in[lane * 128];
          uint32_t at_i = 0;
          const auto fill = [&](uint32_t n, bool is_c) {
            for (uint32_t i = 0; i < n; ++i, ++at_i) {
              switch (f.kind) {
                case Kind::F16:
                  row[at_i] = is_c && !f.c_f16 ? f32_class(g, wide) : half_class(g, wide) | uint32_t{half_class(g, wide)} << 16;
                  break;
                case Kind::BF16: row[at_i] = is_c ? f32_class(g, wide) : (f32_class(g, wide) >> 16) | (f32_class(g, wide) & 0xffff0000u); break;
                case Kind::TF32: row[at_i] = f32_class(g, wide); break;
                case Kind::F64: row[at_i] = f64_class(g, wide, trial >= 6); break;
                case Kind::Int: {
                  const uint32_t h = g();
                  row[at_i] = is_c ? (h % 4 == 0 ? 0x7fffffffu - (h >> 8) % 100000 : h % 4 == 1 ? 0x80000000u + (h >> 8) % 100000 : g()) : (h % 5 == 0 ? 0x80808080u : g());
                  break;
                }
                default: row[at_i] = g();
              }
            }
          };
          fill(ra, false);
          fill(rb, false);
          fill(rc, true);
        }
        const bool partial = trial == 5;   // half the lanes predicated off
        std::vector<uint64_t> fast, slow;
        try {
          setenv("VGPU_FASTPATH", "1", 1);
          fast = run(f, layouts, in, ra, rb, rc, rd, partial ? 16 : 32);
          setenv("VGPU_FASTPATH", "0", 1);
          slow = run(f, layouts, in, ra, rb, rc, rd, partial ? 16 : 32);
        } catch (const Error& e) {
          unsetenv("VGPU_FASTPATH");
          throw vtest::Failure(std::string(f.name) + " " + layouts + " trial " + std::to_string(trial) + ": " + e.what());
        }
        unsetenv("VGPU_FASTPATH");
        for (size_t i = 0; i < fast.size(); ++i) {
          ++compared;
          if (fast[i] != slow[i]) {
            char buf[300];
            std::snprintf(buf, sizeof buf, "%s %s trial %d: word %zu (lane %zu): route 0x%llx, general 0x%llx", f.name, layouts, trial, i, i / rd, static_cast<unsigned long long>(fast[i]),
                          static_cast<unsigned long long>(slow[i]));
            throw vtest::Failure(buf);
          }
        }
      }
    }
  }
  VCHECK(compared > 100000);
}

VTEST_MAIN
