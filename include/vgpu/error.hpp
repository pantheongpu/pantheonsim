// VirtualGPU error model.
//
// One exception type carrying a stable error code (for programmatic handling and
// tests) plus a rich human-readable message. Messages are expected to be useful:
// they should name the object, the operation, and the limit that was violated.
#pragma once

#include <exception>
#include <sstream>
#include <string>

namespace vgpu {

enum class Err {
  UnknownGpu,             // --gpu id not in the registry
  ProfileParse,           // malformed device profile file
  InvalidValue,           // bad argument to an API call
  OutOfMemory,            // virtual VRAM exhausted
  InvalidPointer,         // device pointer was never allocated
  UseAfterFree,           // device pointer inside a freed allocation
  DoubleFree,             // freeing an already-freed allocation
  InvalidFree,            // freeing a pointer that is not an allocation base
  OutOfBounds,            // access past the end of an allocation
  MisalignedAccess,       // address not aligned to access size
  PtxParse,               // syntactically invalid PTX
  UnsupportedPtx,         // valid-looking PTX we do not implement yet
  UninitializedRegister,  // kernel read a register before writing it
  DataRace,               // two warps reached the same shared word unordered
  Trap,                   // kernel executed "trap": failed assert, unreachable path
  LaunchConfig,           // grid/block/shared config violates profile limits
  ExecLimit,              // step budget exceeded (likely infinite loop)
  NotFound,               // module/function lookup failure
  Unsupported,            // unimplemented runtime feature
  Internal,               // invariant violation in the emulator itself
};

const char* err_name(Err e);

class Error : public std::exception {
 public:
  Error(Err code, std::string message) : code_(code), raw_(std::move(message)) {
    msg_ = std::string("VirtualGPU error [") + err_name(code) + "]: " + raw_;
  }

  Err code() const { return code_; }
  const char* what() const noexcept override { return msg_.c_str(); }
  // The message without the "VirtualGPU error [...]" prefix, for rewrapping
  // with extra context at a higher layer.
  const std::string& message() const { return raw_; }

  // Error::make(Err::OutOfBounds, "access at 0x", std::hex, addr, " past end of allocation")
  template <class... Args>
  static Error make(Err code, Args&&... args) {
    std::ostringstream os;
    (os << ... << std::forward<Args>(args));
    return Error(code, os.str());
  }

 private:
  Err code_;
  std::string raw_;
  std::string msg_;
};

}  // namespace vgpu
