/*
 * Copyright 2014-2026 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/**
 * @file cudnn_subquadratic_ops.h
 * @brief cuDNN subquadratic / linear-complexity operations (causal conv1d, etc.).
 *
 * Provides direct function-call APIs for specialized kernels originating from
 * the SubquadraticOps library, without requiring the graph API or engine
 * infrastructure.
 *
 * @note Not supported on Windows.
 *
 * @since cuDNN 9.22.0
 */

#if !defined(CUDNN_SUBQUADRATIC_OPS_H_)
#define CUDNN_SUBQUADRATIC_OPS_H_

#pragma once

#include "cudnn_version.h"
#include "cudnn_ops.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Activation mode for causal conv1d operations.
 *
 * @since cuDNN 9.22.0
 */
typedef enum {
    CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY = 0, /**< Identity (no activation). */
    CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU     = 1, /**< SiLU (Sigmoid Linear Unit) activation. */
} cudnnCausalConv1dActivation_t;

/**
 * @brief Check the version of the cuDNN SubquadraticOps library.
 *
 * Verifies that the SubquadraticOps sub-library version matches the core cuDNN version.
 *
 * @return cudnnStatus_t indicating success or version mismatch.
 *
 * @since cuDNN 9.22.0
 */
cudnnStatus_t CUDNNWINAPI
cudnnSubquadraticOpsVersionCheck(void);

/**
 * @brief Compute a causal (left-padded) depthwise 1D convolution with optional SiLU activation.
 *
 * Computes: y = Act( conv1d_causal(x, weight) + bias )
 *
 * Causal padding inserts (kernel_size - 1) zeros on the left and 0 on the right.
 * The convolution is depthwise: each channel is convolved independently with its
 * own 1D filter.
 *
 * @param[in]  stream      CUDA stream for kernel launch.
 * @param[in]  x           Input tensor in device memory, layout (batch, dim, seq_len), contiguous.
 * @param[in]  weight      Filter tensor in device memory, layout (dim, kernel_size), contiguous.
 * @param[in]  bias        Bias tensor in device memory, layout (dim,), contiguous. Must be non-NULL.
 * @param[out] y           Output tensor in device memory, layout (batch, dim, seq_len), contiguous.
 * @param[in]  batch       Batch size.
 * @param[in]  dim         Number of channels (feature dimension).
 * @param[in]  seqLen      Sequence length.
 * @param[in]  kernelSize  Convolution kernel width. Supported: 2-8, 16, 32, 64, 128, 256.
 * @param[in]  dataType    Element type for x, weight, bias, y. Supported: FLOAT, HALF, BFLOAT16.
 * @param[in]  activation  Activation to apply after convolution + bias.
 *
 * @note Not supported on Windows.
 *
 * @return cudnnStatus_t indicating success or failure.
 *
 * @since cuDNN 9.22.0
 */
cudnnStatus_t CUDNNWINAPI
cudnnCausalConv1dForward(cudaStream_t stream,
                         const void *x,
                         const void *weight,
                         const void *bias,
                         void *y,
                         int batch,
                         int dim,
                         int seqLen,
                         int kernelSize,
                         cudnnDataType_t dataType,
                         cudnnCausalConv1dActivation_t activation);

/**
 * @brief Compute gradients for causal depthwise 1D convolution.
 *
 * Computes:
 * - dx      = dL/dx       (batch, dim, seq_len)
 * - dweight = dL/dweight   (dim, kernel_size) — accumulated via atomicAdd
 * - dbias   = dL/dbias     (dim,)             — accumulated via atomicAdd
 *
 * The caller must zero-initialize dweight and dbias before calling this function
 * if accumulation across multiple calls is not desired.
 *
 * @param[in]     stream      CUDA stream for kernel launch.
 * @param[in]     x           Original input tensor (needed for activation backward), device memory.
 * @param[in]     weight      Original filter tensor in device memory.
 * @param[in]     bias        Original bias tensor in device memory. Must be non-NULL.
 * @param[in]     dy          Output gradient tensor in device memory, layout (batch, dim, seq_len).
 * @param[out]    dx          Input gradient tensor in device memory, layout (batch, dim, seq_len).
 * @param[in,out] dweight     Filter gradient tensor (accumulated) in device memory, layout (dim, kernel_size).
 * @param[in,out] dbias       Bias gradient tensor (accumulated) in device memory, layout (dim,). Must be non-NULL.
 * @param[in]     batch       Batch size.
 * @param[in]     dim         Number of channels.
 * @param[in]     seqLen      Sequence length.
 * @param[in]     kernelSize  Convolution kernel width.
 * @param[in]     dataType    Element type for x, weight, bias, dy, dx. Supported: FLOAT, HALF, BFLOAT16.
 * @param[in]     dwDataType  Element type for dweight, dbias. Currently only FLOAT is supported.
 * @param[in]     activation  Activation that was applied in forward (needed for backward recompute).
 *
 * @note Not supported on Windows.
 *
 * @return cudnnStatus_t indicating success or failure.
 *
 * @since cuDNN 9.22.0
 */
cudnnStatus_t CUDNNWINAPI
cudnnCausalConv1dBackward(cudaStream_t stream,
                          const void *x,
                          const void *weight,
                          const void *bias,
                          const void *dy,
                          void *dx,
                          void *dweight,
                          void *dbias,
                          int batch,
                          int dim,
                          int seqLen,
                          int kernelSize,
                          cudnnDataType_t dataType,
                          cudnnDataType_t dwDataType,
                          cudnnCausalConv1dActivation_t activation);

/*
 * cudnnCausalConv1dNwhForward
 *
 * Same as cudnnCausalConv1dForward but with NWH input/output layout:
 *   x:  (batch, seqLen, dim)
 *   y:  (batch, seqLen, dim)
 * Weight and bias layouts unchanged: (dim, kernelSize) and (dim,).
 */
cudnnStatus_t CUDNNWINAPI
cudnnCausalConv1dNwhForward(cudaStream_t stream,
                            const void *x,
                            const void *weight,
                            const void *bias,
                            void *y,
                            int batch,
                            int dim,
                            int seqLen,
                            int kernelSize,
                            cudnnDataType_t dataType,
                            cudnnCausalConv1dActivation_t activation);

/*
 * cudnnCausalConv1dNwhBackward
 *
 * Same as cudnnCausalConv1dBackward but with NWH input/output layout.
 */
cudnnStatus_t CUDNNWINAPI
cudnnCausalConv1dNwhBackward(cudaStream_t stream,
                             const void *x,
                             const void *weight,
                             const void *bias,
                             const void *dy,
                             void *dx,
                             void *dweight,
                             void *dbias,
                             int batch,
                             int dim,
                             int seqLen,
                             int kernelSize,
                             cudnnDataType_t dataType,
                             cudnnDataType_t dwDataType,
                             cudnnCausalConv1dActivation_t activation);

/*
 * cudnnB2BCausalConv1dForward
 *
 * Fused back-to-back causal conv1d: projection conv → elementwise gating → mixer conv.
 * No activation is applied to individual conv outputs (identity only).
 *
 *   x:            (batch, dim, 3, seqLen)   — 3 interleaved channels per dim
 *   weightsProj:  (dim, 3, kernelSizeProj)  — projection filters for all 3 channels
 *   weightsMixer: (dim, kernelSizeMixer)
 *   skipBias:     (dim,)
 *   y:            (batch, dim, seqLen) — intermediate mixer + skip output.
 *                 Saved and consumed by the backward pass.
 *   yGated:       (batch, dim, seqLen) — final post-gated output = y * projX1.
 *                 This is the primary user-facing output of the fused Hyena-SE block.
 */
cudnnStatus_t CUDNNWINAPI
cudnnB2BCausalConv1dForward(cudaStream_t stream,
                            const void *x,
                            const void *weightsProj,
                            const void *weightsMixer,
                            const void *skipBias,
                            void *y,
                            void *yGated,
                            int batch,
                            int dim,
                            int seqLen,
                            int kernelSizeProj,
                            int kernelSizeMixer,
                            cudnnDataType_t dataType);

/*
 * cudnnB2BCausalConv1dBackward
 *
 * Computes gradients for fused back-to-back causal conv1d.
 *
 *   x:            (batch, dim, 3, seqLen)   — saved input from forward
 *   weightsProj:  (dim, 3, kernelSizeProj)
 *   weightsMixer: (dim, kernelSizeMixer)
 *   skipBias:     (dim,)
 *   y:            (batch, dim, seqLen)      — saved intermediate from forward
 *   dy:           (batch, dim, seqLen)      — gradient of yGated
 *   dx:           (batch, dim, 3, seqLen)   — gradient of x
 *   dweightsProj: (dim, 3, kernelSizeProj)  — gradient of weightsProj (dwDataType)
 *   dweightsMixer:(dim, kernelSizeMixer)    — gradient of weightsMixer (dwDataType)
 *   dskipBias:    (dim,)                    — gradient of skipBias (dwDataType)
 */
cudnnStatus_t CUDNNWINAPI
cudnnB2BCausalConv1dBackward(cudaStream_t stream,
                             const void *x,
                             const void *weightsProj,
                             const void *weightsMixer,
                             const void *skipBias,
                             const void *y,
                             const void *dy,
                             void *dx,
                             void *dweightsProj,
                             void *dweightsMixer,
                             void *dskipBias,
                             int batch,
                             int dim,
                             int seqLen,
                             int kernelSizeProj,
                             int kernelSizeMixer,
                             cudnnDataType_t dataType,
                             cudnnDataType_t dwDataType);

#if defined(__cplusplus)
}
#endif

#endif /* CUDNN_SUBQUADRATIC_OPS_H_ */
