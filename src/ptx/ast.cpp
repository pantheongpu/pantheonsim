#include "vgpu/ptx/ast.hpp"

#include <mutex>
#include <unordered_map>

namespace vgpu::ptx {

std::string Type::str() const {
  if (kind == Kind::Pred) return ".pred";
  // bfloat16 is its own kind, and it was missing here: it fell through to the
  // 'b' default and printed as ".b16", so every diagnostic naming a bf16 type
  // said "bits" instead. The switch is exhaustive now so the compiler catches
  // the next kind that is added.
  if (kind == Kind::BF) return ".bf" + std::to_string(bits);
  char k = 'b';
  switch (kind) {
    case Kind::B: k = 'b'; break;
    case Kind::U: k = 'u'; break;
    case Kind::S: k = 's'; break;
    case Kind::F: k = 'f'; break;
    case Kind::BF: break;   // handled above
    case Kind::Pred: break;
  }
  return std::string(".") + k + std::to_string(bits);
}

// Opcode interning. A process-wide table, because instruction ids outlive the
// module that first mentioned an opcode: a launch counts by id and reports by
// name, and the two must agree across modules.
//
// Guarded by a mutex because modules can be loaded from more than one thread,
// but read on the hot path only through the id already stored in the Instr --
// nothing looks a name up while a kernel is running.
uint16_t intern_opcode(const std::string& mnemonic) {
  static std::mutex mu;
  static std::unordered_map<std::string, uint16_t> ids;
  std::lock_guard<std::mutex> lock(mu);
  auto it = ids.find(mnemonic);
  if (it != ids.end()) return it->second;
  auto& names = const_cast<std::vector<std::string>&>(opcode_names());
  if (names.empty()) names.push_back("<none>");  // id 0 means "not set"
  if (names.size() >= 0xFFFF) return 0;          // absurdly many opcodes; stop growing
  const uint16_t id = static_cast<uint16_t>(names.size());
  names.push_back(mnemonic);
  ids.emplace(mnemonic, id);
  return id;
}

const std::vector<std::string>& opcode_names() {
  static std::vector<std::string> names;
  return names;
}

}  // namespace vgpu::ptx
