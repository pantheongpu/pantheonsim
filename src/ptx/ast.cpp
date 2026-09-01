#include "vgpu/ptx/ast.hpp"

namespace vgpu::ptx {

std::string Type::str() const {
  if (kind == Kind::Pred) return ".pred";
  char k = 'b';
  switch (kind) {
    case Kind::B: k = 'b'; break;
    case Kind::U: k = 'u'; break;
    case Kind::S: k = 's'; break;
    case Kind::F: k = 'f'; break;
    case Kind::Pred: break;
  }
  return std::string(".") + k + std::to_string(bits);
}

}  // namespace vgpu::ptx
