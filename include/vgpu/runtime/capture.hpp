// Graph capture hook for the vendor-library shims.
//
// cuBLAS and its siblings are implemented on the host rather than through the
// SIMT interpreter (see nvidia/docs/cublas.md). That boundary is invisible to a
// program until it captures a CUDA graph: on hardware a library call inside a
// captured region is *recorded* and runs when the graph is launched, so its
// operands are whatever the captured kernels produced. A host implementation
// that instead computed immediately would read operands the graph has not
// produced yet, and would then be absent from every replay.
//
// So a library call made during capture hands its work over as a closure. The
// closure re-reads device memory when the graph runs, which is the behaviour
// the call would have had on a real device.
#ifndef VGPU_RUNTIME_CAPTURE_HPP
#define VGPU_RUNTIME_CAPTURE_HPP

#include <functional>

struct CUstream_st;

// Records `op` when `stream` is capturing and returns true, in which case the
// caller must NOT do the work now -- it belongs to the graph. Returns false
// when there is no capture, and the caller proceeds normally.
bool vgpu_record_host_op_if_capturing(CUstream_st* stream, std::function<void()> op);

#endif  // VGPU_RUNTIME_CAPTURE_HPP
