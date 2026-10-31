// The AST builder, and the finalisation passes it shares with the PTX parser
// (include/vgpu/ptx/builder.hpp).
#include "vgpu/ptx/builder.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/ptx/contract.hpp"
#include "vgpu/ptx/sync_masks.hpp"

namespace vgpu::ptx {

// Dynamic shared memory lives above every static allocation, at the
// alignment its declaration states. Every extern .shared names the same
// bytes, so they share one offset. Starting it at the unaligned end of the
// static data put a TMA destination declared .align 16 at offset 8.
void place_dynamic_shared(EntryFn& fn) {
  uint32_t align = 1;
  for (const auto& [name, d] : fn.shared)
    if (d.dynamic) align = std::max(align, d.align);
  fn.dynamic_shared_offset = (fn.static_shared_size + align - 1) / align * align;
  for (auto& [name, d] : fn.shared)
    if (d.dynamic) d.offset = fn.dynamic_shared_offset;
}

// Point every call at the function it names. Done after the whole module is
// parsed because PTX declares prototypes first and a call can precede the
// definition -- resolving eagerly would refuse a forward reference that is
// perfectly well defined thirty lines later.
void link_calls(Module& m) {
  std::unordered_map<std::string, std::shared_ptr<EntryFn>> by_name;
  for (auto& f : m.funcs) by_name[f->name] = f;
  // `with_tables` is false for the device functions themselves. Giving every
  // EntryFn a list of shared_ptrs to all of them makes each function hold a
  // shared_ptr to itself -- a reference cycle, so no .func is ever freed and
  // every module leaks its whole body. Nothing needs it there anyway: an
  // indirect call resolves against the *kernel's* table, which is the one
  // the interpreter reads.
  auto fix = [&](EntryFn& fn, bool with_tables) {
    if (with_tables) {
      fn.module_funcs.assign(m.funcs.begin(), m.funcs.end());
      fn.module_entry_names.clear();
      for (const auto& e : m.entries) fn.module_entry_names.push_back(e.name);
    }
    for (Instr& ins : fn.body) {
      auto* call = std::get_if<OpCall>(&ins.op);
      if (!call || call->indirect || call->callee.empty()) continue;
      if (call->callee == "vprintf" || call->callee == "__assertfail" ||
          call->callee == "malloc" || call->callee == "free")
        continue;
      // An unresolved name is left unresolved rather than refused here. A
      // separately compiled build links in CUDA's device-runtime library,
      // which declares functions the driver supplies (cnpGetLastError and
      // friends) and defines them nowhere in the module -- so refusing at
      // parse time rejected a whole program because a library function
      // nobody calls had no body. Execution reports it if it is ever
      // reached, which is the point at which it actually matters.
      auto it = by_name.find(call->callee);
      if (it != by_name.end()) call->target = it->second.get();
    }
  };
  for (auto& e : m.entries) fix(e, /*with_tables=*/true);
  for (auto& f : m.funcs) fix(*f, /*with_tables=*/false);
}

void finalize_function(EntryFn& fn, const FinalizeOptions& opt) {
  if (opt.contract_mul_add) contract_mul_add(fn);
  if (opt.nan_order) mark_nan_order(fn);
  if (opt.fold_sync_masks) fold_constant_sync_masks(fn);
  place_dynamic_shared(fn);
}

void finalize_module(Module& m) {
  link_calls(m);
  // Module-scope .shared variables are per-block storage, and a kernel is
  // given a slot for each one it can reach: named in its own body or in the
  // body of a device function it calls, directly or through others. That is
  // what the hardware allocates. Charging every kernel for every one of them
  // refused real programs: a separately compiled build hoists each device
  // function's static __shared__ to module scope, and NanoVDB's 37.9 KB of
  // those plus the device-runtime library's 32.9 KB is more than a T4 holds,
  // so no kernel could be placed. A kernel that makes an indirect call could
  // reach any function, so it keeps them all.
  std::set<std::string> interesting;
  for (const auto& md : m.module_shared) interesting.insert(md.name);
  std::unordered_map<std::string, const EntryFn*> funcs_by_name;
  for (const auto& f : m.funcs) {
    interesting.insert(f->name);
    // A prototype and its definition share a name; the definition has the body.
    auto [it, fresh] = funcs_by_name.emplace(f->name, f.get());
    if (!fresh && it->second->body.empty()) it->second = f.get();
  }
  auto trim = [&](EntryFn& f) {
    for (auto it = f.words_seen.begin(); it != f.words_seen.end();)
      it = interesting.count(*it) ? std::next(it) : f.words_seen.erase(it);
  };
  for (auto& f : m.funcs) trim(*f);
  for (auto& e : m.entries) trim(e);
  auto calls_indirectly = [](const EntryFn& f) {
    for (const Instr& ins : f.body)
      if (const auto* c = std::get_if<OpCall>(&ins.op); c && c->indirect) return true;
    return false;
  };
  // The names a kernel can reach, or nothing when it could reach anything.
  // `called` collects the device functions on the way.
  auto reachable = [&](const EntryFn& e, std::vector<const EntryFn*>& called)
      -> std::optional<std::set<std::string>> {
    if (calls_indirectly(e)) return std::nullopt;
    std::set<std::string> names = e.words_seen;
    std::vector<const EntryFn*> todo;
    std::set<const EntryFn*> done;
    for (const auto& w : e.words_seen)
      if (auto it = funcs_by_name.find(w); it != funcs_by_name.end()) todo.push_back(it->second);
    while (!todo.empty()) {
      const EntryFn* f = todo.back();
      todo.pop_back();
      if (!done.insert(f).second) continue;
      if (calls_indirectly(*f)) return std::nullopt;
      called.push_back(f);
      for (const auto& w : f->words_seen) {
        names.insert(w);
        if (auto it = funcs_by_name.find(w); it != funcs_by_name.end()) todo.push_back(it->second);
      }
    }
    return names;
  };
  for (auto& fn : m.entries) {
    std::vector<const EntryFn*> called;
    const auto reach = reachable(fn, called);
    // A device function's own .shared (declared inside its body, as a
    // separately compiled build emits a function's static __shared__) is
    // resolved in the running kernel's frame, so the kernel needs a slot for
    // it. It used to have none: the callee's first access was "unknown
    // symbol". With an indirect call in reach, every function counts.
    if (!reach) {
      called.clear();
      for (const auto& f : m.funcs) called.push_back(f.get());
    }
    for (const EntryFn* f : called) {
      for (const auto& [name, fd] : f->shared) {
        if (fn.shared.count(name)) continue;
        SharedDecl d = fd;
        if (d.dynamic) {
          fn.uses_dynamic_shared = true;
          d.offset = fn.static_shared_size;
        } else {
          const uint32_t off = (fn.static_shared_size + d.align - 1) / d.align * d.align;
          d.offset = off;
          fn.static_shared_size = off + d.size;
        }
        fn.shared.emplace(name, d);
      }
    }
    for (const auto& md : m.module_shared) {
      if (fn.shared.count(md.name)) continue;
      // Only static .shared costs a kernel anything. A dynamic array
      // (.extern, no size) is the launch's dynamic allocation under another
      // name, so every kernel has it, whether or not it names it.
      if (reach && !reach->count(md.name) && !md.dynamic) continue;
      SharedDecl d = md;
      if (d.dynamic) {
        fn.uses_dynamic_shared = true;
        d.offset = fn.static_shared_size;
      } else {
        uint32_t off = (fn.static_shared_size + d.align - 1) / d.align * d.align;
        d.offset = off;
        fn.static_shared_size = off + d.size;
      }
      fn.shared.emplace(d.name, d);
    }
    place_dynamic_shared(fn);
  }
  // Last, once nothing edits an instruction any more (mul/add contraction,
  // sync-mask folding and call resolution all do): the pre-decoded forms must
  // describe the body the interpreter will run.
  intern_call_slots(m);
  for (auto& fn : m.entries) build_micro(fn);
  for (auto& fn : m.funcs) build_micro(*fn);
}

// ---------------------------------------------------------------------------
// KernelBuilder

namespace {

uint32_t align_up(uint32_t v, uint32_t a) { return a ? (v + a - 1) / a * a : v; }

[[noreturn]] void builder_error(const std::string& fn, const std::string& why) {
  throw Error::make(Err::InvalidValue, "AST builder, function '", fn, "': ", why);
}

}  // namespace

KernelBuilder::KernelBuilder(std::string name, bool device_func) {
  fn_.name = std::move(name);
  fn_.is_device_func = device_func;
}

Reg KernelBuilder::reg(const std::string& name, Type ty) {
  const bool wide = ty.bits > 32 && ty.kind != Type::Kind::Pred;
  if (fn_.reg_ids.count(name)) {
    auto was = fn_.reg_wide.find(name);
    const bool was_wide = was != fn_.reg_wide.end() && was->second;
    if (was_wide != wide)
      builder_error(fn_.name, "register '" + name + "' is declared " + ty.str() + " but was already declared " +
                                  (was_wide ? "64-bit" : "32-bit"));
  }
  fn_.reg_decls[name] = ty;
  fn_.reg_wide[name] = wide;
  auto it = fn_.reg_ids.find(name);
  if (it == fn_.reg_ids.end()) {
    const uint32_t id = wide ? fn_.num_regs64++ : fn_.num_regs32++;
    it = fn_.reg_ids.emplace(name, id).first;
    ++fn_.num_regs;
  }
  return Reg{name, it->second, wide};
}

Reg KernelBuilder::temp(Type ty) {
  std::string name;
  do {
    name = "%b" + std::to_string(temps_++);
  } while (fn_.reg_ids.count(name));
  return reg(name, ty);
}

Reg KernelBuilder::pred() { return temp(Type{Type::Kind::Pred, 1}); }

size_t KernelBuilder::add_param(const std::string& name, Type ty, uint32_t size, uint32_t align, bool local_mem) {
  if (fn_.is_device_func) builder_error(fn_.name, "a device function's parameters are call slots (add_call_param)");
  ParamDecl p;
  p.name = name;
  p.ty = ty;
  p.size = size ? size : ty.bytes();
  p.align = align ? align : (p.size < 8 ? p.size : 8);
  p.local_mem = local_mem;
  if (local_mem && p.size != 4 && p.size != 8)
    builder_error(fn_.name, "local-memory parameter '" + name + "' must be 4 or 8 bytes, not " + std::to_string(p.size));
  fn_.params.push_back(std::move(p));
  return fn_.params.size() - 1;
}

Addr KernelBuilder::param_addr(size_t index, int64_t offset) const {
  if (index >= fn_.params.size()) builder_error(fn_.name, "parameter " + std::to_string(index) + " does not exist");
  Addr a;
  a.base_kind = Addr::Base::EntryParam;
  a.base = fn_.params[index].name;
  a.param_index = static_cast<uint32_t>(index);
  a.offset = offset;
  return a;
}

void KernelBuilder::add_call_param(const std::string& slot, uint32_t bytes) {
  fn_.param_slot_names.push_back(slot);
  fn_.param_slot_bytes.push_back(bytes);
}

void KernelBuilder::set_return(const std::string& slot, uint32_t bytes) {
  fn_.retval_slot_name = slot;
  fn_.retval_bytes = bytes;
}

Addr KernelBuilder::slot_addr(const std::string& slot, int64_t offset) {
  Addr a;
  a.base_kind = Addr::Base::CallSlot;
  a.base = slot;
  a.offset = offset;
  return a;
}

uint32_t KernelBuilder::declare_shared(const std::string& name, uint32_t size, uint32_t align) {
  SharedDecl d;
  d.name = name;
  d.size = size;
  d.align = align ? align : 8;
  const uint64_t off = align_up(fn_.static_shared_size, d.align);
  if (off + size > UINT32_MAX) builder_error(fn_.name, ".shared memory is too large");
  d.offset = static_cast<uint32_t>(off);
  fn_.static_shared_size = static_cast<uint32_t>(off + size);
  if (!fn_.shared.emplace(name, d).second) builder_error(fn_.name, "duplicate .shared '" + name + "'");
  return d.offset;
}

void KernelBuilder::declare_dynamic_shared(const std::string& name, uint32_t align) {
  SharedDecl d;
  d.name = name;
  d.size = 0;
  d.align = align ? align : 8;
  d.dynamic = true;
  fn_.uses_dynamic_shared = true;
  if (!fn_.shared.emplace(name, d).second) builder_error(fn_.name, "duplicate .shared '" + name + "'");
}

uint32_t KernelBuilder::declare_local(const std::string& name, uint32_t size, uint32_t align) {
  LocalDecl d;
  d.name = name;
  d.size = size;
  d.align = align ? align : 8;
  const uint64_t off = align_up(fn_.local_frame_size, d.align);
  if (off + size > UINT32_MAX) builder_error(fn_.name, ".local frame is too large");
  d.offset = static_cast<uint32_t>(off);
  fn_.local_frame_size = static_cast<uint32_t>(off + size);
  if (!fn_.locals.emplace(name, d).second) builder_error(fn_.name, "duplicate .local '" + name + "'");
  return d.offset;
}

Addr KernelBuilder::symbol_addr(const std::string& name, int64_t offset) {
  Addr a;
  a.base_kind = Addr::Base::Symbol;
  a.base = name;
  a.offset = offset;
  return a;
}

Addr KernelBuilder::reg_addr(const Reg& base, int64_t offset) {
  Addr a;
  a.base_kind = Addr::Base::Reg;
  a.base = base.name;
  a.base_id = base.id;
  a.base_wide = base.wide;
  a.offset = offset;
  return a;
}

void KernelBuilder::mention(const std::string& name) { fn_.words_seen.insert(name); }

Instr KernelBuilder::make_instr(Op op, const std::string& mnemonic, const std::string& text, size_t line) const {
  Instr ins;
  const std::string& m = mnemonic.empty() ? std::string("op") : mnemonic;
  ins.opcode_id = intern_opcode(m);
  ins.text = text.empty() ? m : text;
  ins.line = line ? line : fn_.body.size() + 1;
  ins.op = std::move(op);
  return ins;
}

size_t KernelBuilder::emit(Op op, const std::string& mnemonic, const std::string& text, size_t line) {
  if (finished_) builder_error(fn_.name, "the builder is spent (finish() was called)");
  // The names a body mentions decide which module-scope .shared variables and
  // device functions a kernel reaches (see KernelBuilder::mention).
  auto addr_name = [&](const Addr& a) {
    if (a.base_kind == Addr::Base::Symbol) fn_.words_seen.insert(a.base);
  };
  if (const auto* o = std::get_if<OpLd>(&op)) addr_name(o->addr);
  else if (const auto* o2 = std::get_if<OpSt>(&op)) addr_name(o2->addr);
  else if (const auto* o3 = std::get_if<OpAtom>(&op)) addr_name(o3->addr);
  else if (const auto* o4 = std::get_if<OpMov>(&op)) {
    if (const auto* sy = std::get_if<SymbolOperand>(&o4->src)) fn_.words_seen.insert(sy->name);
  } else if (const auto* o5 = std::get_if<OpCall>(&op)) {
    if (!o5->callee.empty()) fn_.words_seen.insert(o5->callee);
  }
  fn_.body.push_back(make_instr(std::move(op), mnemonic, text, line));
  return fn_.body.size() - 1;
}

size_t KernelBuilder::emit_if(const Reg& guard, bool negate, Op op, const std::string& mnemonic,
                              const std::string& text, size_t line) {
  const size_t at = emit(std::move(op), mnemonic, text, line);
  Instr& ins = fn_.body[at];
  ins.has_pred = true;
  ins.pred_negated = negate;
  ins.pred = guard;
  return at;
}

Label KernelBuilder::make_label() {
  labels_.emplace_back();
  return Label{labels_.size() - 1};
}

void KernelBuilder::bind(Label l) {
  if (l.id >= labels_.size()) builder_error(fn_.name, "bind of a label that is not this builder's");
  if (labels_[l.id].bound) builder_error(fn_.name, "label L" + std::to_string(l.id) + " is bound twice");
  labels_[l.id].bound = true;
  labels_[l.id].pos = fn_.body.size();
}

size_t KernelBuilder::bra(Label target, const std::optional<Reg>& guard, bool negate) {
  if (target.id >= labels_.size()) builder_error(fn_.name, "bra to a label that is not this builder's");
  OpBra b;
  b.label = "L" + std::to_string(target.id);
  const size_t at = guard ? emit_if(*guard, negate, std::move(b), "bra") : emit(std::move(b), "bra");
  bra_fixups_.emplace_back(at, target);
  return at;
}

size_t KernelBuilder::brx(const Operand& index, const std::vector<Label>& targets) {
  OpBrx b;
  b.a = index;
  b.table = "brx" + std::to_string(brx_fixups_.size());
  for (const Label& t : targets) {
    if (t.id >= labels_.size()) builder_error(fn_.name, "brx to a label that is not this builder's");
    b.labels.push_back("L" + std::to_string(t.id));
  }
  const size_t at = emit(std::move(b), "brx");
  brx_fixups_.emplace_back(at, targets);
  return at;
}

EntryFn KernelBuilder::finish(const FinalizeOptions& opt) {
  if (finished_) builder_error(fn_.name, "finish() was called twice");
  auto resolve = [&](Label l) {
    if (!labels_[l.id].bound) builder_error(fn_.name, "label L" + std::to_string(l.id) + " is branched to but never bound");
    return labels_[l.id].pos;
  };
  for (const auto& [at, l] : bra_fixups_) std::get<OpBra>(fn_.body[at].op).target = resolve(l);
  for (const auto& [at, ls] : brx_fixups_) {
    auto& b = std::get<OpBrx>(fn_.body[at].op);
    b.targets.clear();
    for (const Label& l : ls) b.targets.push_back(resolve(l));
  }
  finalize_function(fn_, opt);
  finished_ = true;
  return std::move(fn_);
}

// ---------------------------------------------------------------------------
// ModuleBuilder

size_t ModuleBuilder::add_entry(EntryFn fn) {
  if (finished_) throw Error::make(Err::InvalidValue, "AST builder: the module builder is spent");
  if (fn.is_device_func) builder_error(fn.name, "a device function is added with add_func");
  m_.entries.push_back(std::move(fn));
  return m_.entries.size() - 1;
}

size_t ModuleBuilder::add_func(EntryFn fn) {
  if (finished_) throw Error::make(Err::InvalidValue, "AST builder: the module builder is spent");
  if (!fn.is_device_func) builder_error(fn.name, "a kernel is added with add_entry");
  m_.funcs.push_back(std::make_shared<EntryFn>(std::move(fn)));
  return m_.funcs.size() - 1;
}

void ModuleBuilder::add_global(GlobalVar g) {
  if (g.size == 0) builder_error(g.name, "a zero-sized global");
  if (!g.init.empty() && g.init.size() != g.size)
    builder_error(g.name, "a global's initialiser must be empty or exactly its " + std::to_string(g.size) + " bytes");
  m_.globals.push_back(std::move(g));
}

void ModuleBuilder::add_module_shared(SharedDecl d) { m_.module_shared.push_back(std::move(d)); }

Module ModuleBuilder::finish() {
  if (finished_) throw Error::make(Err::InvalidValue, "AST builder: finish() was called twice");
  finalize_module(m_);
  finished_ = true;
  return std::move(m_);
}

}  // namespace vgpu::ptx
