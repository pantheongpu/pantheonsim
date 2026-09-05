#include "vgpu/error.hpp"

namespace vgpu {

const char* err_name(Err e) {
  switch (e) {
    case Err::UnknownGpu: return "unknown-gpu";
    case Err::ProfileParse: return "profile-parse";
    case Err::InvalidValue: return "invalid-value";
    case Err::OutOfMemory: return "out-of-memory";
    case Err::InvalidPointer: return "invalid-pointer";
    case Err::UseAfterFree: return "use-after-free";
    case Err::DoubleFree: return "double-free";
    case Err::InvalidFree: return "invalid-free";
    case Err::OutOfBounds: return "out-of-bounds";
    case Err::MisalignedAccess: return "misaligned-access";
    case Err::PtxParse: return "ptx-parse";
    case Err::UnsupportedPtx: return "unsupported-ptx";
    case Err::UninitializedRegister: return "uninitialized-register";
    case Err::DataRace: return "data-race";
    case Err::LaunchConfig: return "launch-config";
    case Err::ExecLimit: return "exec-limit";
    case Err::NotFound: return "not-found";
    case Err::Unsupported: return "unsupported";
    case Err::Internal: return "internal";
  }
  return "unknown";
}

}  // namespace vgpu
