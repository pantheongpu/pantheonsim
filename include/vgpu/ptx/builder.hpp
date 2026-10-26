// The AST builder: how a frontend that does not read PTX text (a SPIR-V
// lowering, a test) makes a kernel the shared interpreter can run.
//
// The PTX parser always did three things besides turning text into Instrs, and
// all three are needed by anything else that produces a kernel:
//
//   1. register interning -- every register gets a dense id in the 32-bit or
//      the 64-bit file (a predicate is a narrow register), and the kernel
//      records how many of each;
//   2. layout -- static and dynamic .shared placement, the per-thread .local
//      frame, the parameter list, call slots;
//   3. finalisation -- the analysis passes that rewrite or annotate the body
//      (NaN operand order, constant member masks), the numbering of call slots
//      and the pre-decoded MicroOps the interpreter's fast paths run.
//
// This header exposes them. The PTX parser is written on top of the same
// functions (finalize_function, finalize_module), so what the builder makes and
// what the parser makes go through one code path.
//
// What the builder does NOT do is the one PTX-only rewrite: mul/add
// contraction (contract_mul_add). That is ptxas's code-generation rule for
// `mul` followed by `add` with no rounding modifier; a frontend whose source
// language says when a multiply and an add may fuse (SPIR-V: the ContractionOff
// execution mode and the FPFastMathMode decorations) emits an Fma itself when
// it means one, and must not have the engine decide for it.
//
// Typical use:
//
//   ptx::KernelBuilder kb("vadd");
//   const size_t pa = kb.add_param("a", {Type::Kind::U, 64}), ...;
//   Reg tid = kb.reg("%tid", {Type::Kind::U, 32});
//   Reg a = kb.reg("%a", {Type::Kind::U, 64});
//   kb.emit(ptx::OpLd{Space::Param, {Type::Kind::U, 64}, {a}, kb.param_addr(pa)}, "ld");
//   ...
//   ptx::ModuleBuilder mb;
//   mb.add_entry(kb.finish());
//   uint64_t module = device.load_built_module(mb.finish());
//
// Nothing here is thread-safe; a builder belongs to the thread that makes it.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vgpu/ptx/ast.hpp"

namespace vgpu::ptx {

// Which passes finalize_function() runs on a function body. The defaults are
// the ones the builder uses. The PTX parser asks for contraction as well.
struct FinalizeOptions {
  // PTX only: rewrite a mul and the add/sub that consumes its product into an
  // fma, as ptxas does (src/ptx/contract.cpp). Off for every other frontend.
  bool contract_mul_add = false;
  // Record which of two NaN operands an f64 add/mul/min/max/fma returns
  // (mark_nan_order). Independent of contraction.
  bool nan_order = true;
  // Replace the member-mask register of a *.sync instruction that is set once
  // from a constant by the constant (fold_constant_sync_masks).
  bool fold_sync_masks = true;
};

// The passes that follow a complete body, in the order the parser has always
// run them: contraction (if asked), NaN order, constant member masks, then the
// placement of dynamic .shared above the static allocations.
void finalize_function(EntryFn& fn, const FinalizeOptions& opt = {});

// Places a function's dynamic .shared (the `extern .shared` form whose size is
// the launch's) above every static allocation, at the largest alignment any
// such declaration asks for. Run by finalize_function and by finalize_module.
void place_dynamic_shared(EntryFn& fn);

// Points every OpCall at the Module::funcs entry it names and gives each kernel
// the tables an indirect call and a "which names are kernels" diagnostic read
// (module_funcs, module_entry_names). A call to a name the module does not
// define is left unresolved, to be reported if it is ever executed.
void link_calls(Module& m);

// Everything that has to happen once a Module holds all its functions:
// link_calls, the attribution of module-scope .shared to the kernels that reach
// it, placement of dynamic shared, call-slot numbering and the MicroOp decode of
// every body. The parser ends with it; ModuleBuilder::finish() calls it.
void finalize_module(Module& m);

// ---------------------------------------------------------------------------

// A position in a function body that branches can name before it is known.
struct Label {
  size_t id = SIZE_MAX;
};

// Builds one EntryFn: a kernel (.entry) or, with `device_func`, a function
// other kernels call.
class KernelBuilder {
 public:
  explicit KernelBuilder(std::string name, bool device_func = false);

  // ---- registers ----------------------------------------------------------
  //
  // Declares a virtual register and returns it. A register wider than 32 bits
  // lives in the 64-bit file; everything else, a predicate included, in the
  // 32-bit one. Declaring a name again with the same width returns the same
  // register (and updates its recorded type); declaring it at the other width
  // throws, as the parser refuses it. Names are for diagnostics and need only
  // be unique within the function.
  Reg reg(const std::string& name, Type ty);
  // A fresh register with a generated name ("%b<n>"), for a value that has no
  // name of its own.
  Reg temp(Type ty);
  // A fresh predicate register.
  Reg pred();

  // ---- parameters ---------------------------------------------------------
  //
  // A kernel parameter. `size` 0 means ty.bytes(); `align` 0 means the natural
  // alignment (the size, at most 8). Returns the parameter's position, which
  // param_addr() and the launch's argument order use. `local_mem`: see
  // ParamDecl::local_mem.
  size_t add_param(const std::string& name, Type ty, uint32_t size = 0, uint32_t align = 0, bool local_mem = false);
  // The address operand of a load from kernel parameter `index`, `offset`
  // bytes into it (ld.param).
  Addr param_addr(size_t index, int64_t offset = 0) const;

  // ---- device functions ---------------------------------------------------
  //
  // A device function's parameters and return value are call slots (each lane
  // passes its own), not a launch buffer. Slot names are what OpStSlot /
  // OpLdSlot / OpDeclSlot / OpCall name.
  void add_call_param(const std::string& slot, uint32_t bytes);
  void set_return(const std::string& slot, uint32_t bytes);
  // An address operand naming a call slot, `offset` bytes in.
  static Addr slot_addr(const std::string& slot, int64_t offset = 0);

  // ---- memory -------------------------------------------------------------
  //
  // Static per-block shared memory. Returns the variable's offset in the
  // block's shared window. Throws on a duplicate name.
  uint32_t declare_shared(const std::string& name, uint32_t size, uint32_t align = 8);
  // The launch's dynamic shared memory under a name (the `extern .shared` form).
  // Every dynamic declaration names the same bytes; they start above the static
  // ones (place_dynamic_shared).
  void declare_dynamic_shared(const std::string& name, uint32_t align = 8);
  // Per-thread local memory: a frame slice of `size` bytes. Returns its offset
  // in the thread's frame.
  uint32_t declare_local(const std::string& name, uint32_t size, uint32_t align = 8);
  // The address operand of a variable named directly ([name + offset]): a
  // .shared or .local variable, or a module global.
  static Addr symbol_addr(const std::string& name, int64_t offset = 0);
  // The address operand [reg + offset].
  static Addr reg_addr(const Reg& base, int64_t offset = 0);

  // A name the body refers to that is a module-scope .shared variable or a
  // device function. The module-level finalisation charges a kernel only for
  // the module .shared it can reach, and reach is judged from these names:
  // call mention() for every module-scope symbol or callee the body uses.
  // (A body that makes an indirect call reaches everything without it.)
  void mention(const std::string& name);

  // ---- instructions -------------------------------------------------------
  //
  // Appends an instruction and returns its index. `mnemonic` is the base opcode
  // name the per-opcode instruction histogram counts under ("add", "ld"; empty:
  // "op"); `text` is the diagnostic text an error names (empty: the mnemonic).
  // `line` is the source line a diagnostic reports (0: the instruction's own
  // 1-based position).
  size_t emit(Op op, const std::string& mnemonic = {}, const std::string& text = {}, size_t line = 0);
  // The same, guarded by a predicate: @p / @!p.
  size_t emit_if(const Reg& guard, bool negate, Op op, const std::string& mnemonic = {}, const std::string& text = {},
                 size_t line = 0);

  // The index the next instruction will have.
  size_t pc() const { return fn_.body.size(); }

  // ---- control flow -------------------------------------------------------
  Label make_label();
  // Binds the label to the next instruction to be emitted. Throws if already bound.
  void bind(Label l);
  // bra to a label; guarded when `guard` is given.
  size_t bra(Label target, const std::optional<Reg>& guard = std::nullopt, bool negate = false);
  // brx.idx: `index` selects an entry of `targets` per lane.
  size_t brx(const Operand& index, const std::vector<Label>& targets);

  // ---- launch bounds ------------------------------------------------------
  void set_req_ntid(uint32_t x, uint32_t y, uint32_t z) { fn_.req_ntid = {x, y, z}; }
  void set_max_ntid(uint32_t x, uint32_t y, uint32_t z) { fn_.max_ntid = {x, y, z}; }
  void set_req_subgroups(uint32_t n) { fn_.req_subgroups = n; }

  // The function under construction, for the fields that have no setter (a
  // frontend that knows better than this class is welcome to). Valid until finish().
  EntryFn& function() { return fn_; }
  const EntryFn& function() const { return fn_; }

  // Resolves the labels and runs finalize_function(). Throws Err::InvalidValue for
  // a label that was branched to and never bound. The builder is spent afterwards.
  EntryFn finish(const FinalizeOptions& opt = {});

 private:
  struct LabelInfo {
    size_t pos = SIZE_MAX;
    bool bound = false;
  };
  Instr make_instr(Op op, const std::string& mnemonic, const std::string& text, size_t line) const;

  EntryFn fn_;
  std::vector<LabelInfo> labels_;
  std::vector<std::pair<size_t, Label>> bra_fixups_;                      // instruction index, target
  std::vector<std::pair<size_t, std::vector<Label>>> brx_fixups_;
  uint32_t temps_ = 0;
  bool finished_ = false;
};

// Assembles a Module from finished functions and the module's data.
class ModuleBuilder {
 public:
  // The .target a module reports to cudaFuncGetAttributes' ptxVersion and to
  // Device::module_arch. Empty (the default) says the module names none.
  void set_target(std::string target) { m_.target = std::move(target); }
  // Adds a finished kernel / device function; returns its position in
  // Module::entries / Module::funcs.
  size_t add_entry(EntryFn fn);
  size_t add_func(EntryFn fn);
  // A module-scope variable, materialised into device memory when the module is
  // loaded: constant data goes in GlobalVar::init (empty: zeros), `is_const`
  // marks module constant memory.
  void add_global(GlobalVar g);
  // A module-scope .shared variable; each kernel that mentions it (or a function
  // that does) is charged for it.
  void add_module_shared(SharedDecl d);
  // Runs finalize_module() and returns the module. The builder is spent.
  Module finish();

 private:
  Module m_;
  bool finished_ = false;
};

}  // namespace vgpu::ptx
