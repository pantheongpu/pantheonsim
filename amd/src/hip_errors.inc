// Every hipError_t of ROCm 7.1's hip_runtime_api.h, by value, with its name
// and the text hipGetErrorString gives: E(value, name, text). Where two names
// share a value (hipErrorMemoryAllocation, hipErrorInitializationError,
// hipErrorMapBufferObjectFailed, hipErrorInvalidResourceHandle), the first the
// header lists is the one given back.
//
// The texts are ROCm's HIP runtime's, as AMD's hip-tests
// (catch/unit/errorHandling/error_handling_common.cc) expect them, which is
// distributed under the MIT licence below; the few that file does not list
// are VirtualGPU's own.
//
// Copyright (c) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

E(0, "hipSuccess", "no error")
E(1, "hipErrorInvalidValue", "invalid argument")
E(2, "hipErrorOutOfMemory", "out of memory")
E(3, "hipErrorNotInitialized", "initialization error")
E(4, "hipErrorDeinitialized", "driver shutting down")
E(5, "hipErrorProfilerDisabled", "profiler disabled while using external profiling tool")
E(6, "hipErrorProfilerNotInitialized", "profiler is not initialized")
E(7, "hipErrorProfilerAlreadyStarted", "profiler already started")
E(8, "hipErrorProfilerAlreadyStopped", "profiler already stopped")
E(9, "hipErrorInvalidConfiguration", "invalid configuration argument")
E(12, "hipErrorInvalidPitchValue", "invalid pitch argument")
E(13, "hipErrorInvalidSymbol", "invalid device symbol")
E(17, "hipErrorInvalidDevicePointer", "invalid device pointer")
E(21, "hipErrorInvalidMemcpyDirection", "invalid copy direction for memcpy")
E(35, "hipErrorInsufficientDriver", "driver version is insufficient for runtime version")
E(52, "hipErrorMissingConfiguration", "__global__ function call is not configured")
E(53, "hipErrorPriorLaunchFailure", "unspecified launch failure in prior launch")
E(98, "hipErrorInvalidDeviceFunction", "invalid device function")
E(100, "hipErrorNoDevice", "no ROCm-capable device is detected")
E(101, "hipErrorInvalidDevice", "invalid device ordinal")
E(200, "hipErrorInvalidImage", "device kernel image is invalid")
E(201, "hipErrorInvalidContext", "invalid device context")
E(202, "hipErrorContextAlreadyCurrent", "context is already current context")
E(205, "hipErrorMapFailed", "mapping of buffer object failed")
E(206, "hipErrorUnmapFailed", "unmapping of buffer object failed")
E(207, "hipErrorArrayIsMapped", "array is mapped")
E(208, "hipErrorAlreadyMapped", "resource already mapped")
E(209, "hipErrorNoBinaryForGpu", "no kernel image is available for execution on the device")
E(210, "hipErrorAlreadyAcquired", "resource already acquired")
E(211, "hipErrorNotMapped", "resource not mapped")
E(212, "hipErrorNotMappedAsArray", "resource not mapped as array")
E(213, "hipErrorNotMappedAsPointer", "resource not mapped as pointer")
E(214, "hipErrorECCNotCorrectable", "uncorrectable ECC error encountered")
E(215, "hipErrorUnsupportedLimit", "limit is not supported on this architecture")
E(216, "hipErrorContextAlreadyInUse", "exclusive-thread device already in use by a different thread")
E(217, "hipErrorPeerAccessUnsupported", "peer access is not supported between these two devices")
E(218, "hipErrorInvalidKernelFile", "invalid kernel file")
E(219, "hipErrorInvalidGraphicsContext", "invalid OpenGL or DirectX context")
E(300, "hipErrorInvalidSource", "device kernel image is invalid")
E(301, "hipErrorFileNotFound", "file not found")
E(302, "hipErrorSharedObjectSymbolNotFound", "shared object symbol not found")
E(303, "hipErrorSharedObjectInitFailed", "shared object initialization failed")
E(304, "hipErrorOperatingSystem", "OS call failed or operation not supported on this OS")
E(400, "hipErrorInvalidHandle", "invalid resource handle")
E(401, "hipErrorIllegalState", "the operation cannot be performed in the present state")
E(500, "hipErrorNotFound", "named symbol not found")
E(600, "hipErrorNotReady", "device not ready")
E(700, "hipErrorIllegalAddress", "an illegal memory access was encountered")
E(701, "hipErrorLaunchOutOfResources", "too many resources requested for launch")
E(702, "hipErrorLaunchTimeOut", "the launch timed out and was terminated")
E(704, "hipErrorPeerAccessAlreadyEnabled", "peer access is already enabled")
E(705, "hipErrorPeerAccessNotEnabled", "peer access has not been enabled")
E(708, "hipErrorSetOnActiveProcess", "cannot set while device is active in this process")
E(709, "hipErrorContextIsDestroyed", "context is destroyed")
E(710, "hipErrorAssert", "device-side assert triggered")
E(712, "hipErrorHostMemoryAlreadyRegistered", "part or all of the requested memory range is already mapped")
E(713, "hipErrorHostMemoryNotRegistered", "pointer does not correspond to a registered memory region")
E(719, "hipErrorLaunchFailure", "unspecified launch failure")
E(720, "hipErrorCooperativeLaunchTooLarge", "too many blocks in cooperative launch")
E(801, "hipErrorNotSupported", "operation not supported")
E(900, "hipErrorStreamCaptureUnsupported", "operation not permitted when stream is capturing")
E(901, "hipErrorStreamCaptureInvalidated", "operation failed due to a previous error during capture")
E(902, "hipErrorStreamCaptureMerge", "operation would result in a merge of separate capture sequences")
E(903, "hipErrorStreamCaptureUnmatched", "capture was not ended in the same stream as it began")
E(904, "hipErrorStreamCaptureUnjoined", "capturing stream has unjoined work")
E(905, "hipErrorStreamCaptureIsolation", "dependency created on uncaptured work in another stream")
E(906, "hipErrorStreamCaptureImplicit", "operation would make the legacy stream depend on a capturing blocking stream")
E(907, "hipErrorCapturedEvent", "operation not permitted on an event last recorded in a capturing stream")
E(908, "hipErrorStreamCaptureWrongThread", "attempt to terminate a thread-local capture sequence from another thread")
E(910, "hipErrorGraphExecUpdateFailure", "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update")
E(911, "hipErrorInvalidChannelDescriptor", "invalid channel descriptor")
E(912, "hipErrorInvalidTexture", "invalid texture")
E(999, "hipErrorUnknown", "unknown error")
E(1052, "hipErrorRuntimeMemory", "runtime memory call returned error")
E(1053, "hipErrorRuntimeOther", "runtime call other than memory returned error")
E(1054, "hipErrorTbd", "unknown error")
