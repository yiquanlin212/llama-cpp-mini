#pragma once

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "llamacpp/tensor.h"

namespace llamacpp {

// ============================================================
// matmul: C[M,N] = A[M,K] @ B[K,N]
//
// Loop order is i-k-j (rank-1 update), NOT the textbook i-j-k.
// This matters because A, B, C are all row-major:
//   - innermost j loop touches B[k,:] and C[i,:] contiguously
//   - cache lines are reused for the entire j sweep
//   - clang -O3 can auto-vectorize the inner loop with NEON
// Empirically 5-15x faster than i-j-k on Apple Silicon.
// Step 7 will replace with hand-written NEON intrinsics + tiling.
// ============================================================
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("matmul: both inputs must be 2D");
    }
    int64_t M = a.shape()[0];
    int64_t K = a.shape()[1];
    int64_t K2 = b.shape()[0];
    int64_t N = b.shape()[1];
    if (K != K2) {
        throw std::runtime_error("matmul: inner dims mismatch");
    }

    Tensor c = Tensor::zeros({M, N});
    const float* ap = a.data();
    const float* bp = b.data();
    float*       cp = c.data();

    // Cache-friendly i-k-j order
    for (int64_t i = 0; i < M; ++i) {
        float* c_row = cp + i * N;
        for (int64_t k = 0; k < K; ++k) {
            float a_ik = ap[i * K + k];
            const float* b_row = bp + k * N;
            for (int64_t j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    return c;
}

// ============================================================
// softmax_inplace: applied along the LAST dimension.
// Numerically stable: subtracts max before exp to avoid overflow.
// ============================================================
inline void softmax_inplace(Tensor& x) {
    int64_t last_dim = x.shape().back();
    int64_t outer    = x.numel() / last_dim;
    float*  p        = x.data();

    for (int64_t i = 0; i < outer; ++i) {
        float* row = p + i * last_dim;

        float max_val = row[0];
        for (int64_t j = 1; j < last_dim; ++j) {
            if (row[j] > max_val) max_val = row[j];
        }

        float sum = 0.0f;
        for (int64_t j = 0; j < last_dim; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum   += row[j];
        }

        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < last_dim; ++j) {
            row[j] *= inv_sum;
        }
    }
}

// ============================================================
// RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
// ============================================================
inline Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-5f) {
    if (weight.ndim() != 1) {
        throw std::runtime_error("rmsnorm: weight must be 1D");
    }
    int64_t last_dim = x.shape().back();
    if (weight.shape()[0] != last_dim) {
        throw std::runtime_error("rmsnorm: weight size mismatch");
    }
    int64_t outer = x.numel() / last_dim;

    Tensor y(x.shape());
    const float* xp = x.data();
    const float* wp = weight.data();
    float*       yp = y.data();

    for (int64_t i = 0; i < outer; ++i) {
        const float* xrow = xp + i * last_dim;
        float*       yrow = yp + i * last_dim;

        float sum_sq = 0.0f;
        for (int64_t j = 0; j < last_dim; ++j) {
            sum_sq += xrow[j] * xrow[j];
        }
        float rms     = std::sqrt(sum_sq / static_cast<float>(last_dim) + eps);
        float inv_rms = 1.0f / rms;

        for (int64_t j = 0; j < last_dim; ++j) {
            yrow[j] = xrow[j] * inv_rms * wp[j];
        }
    }
    return y;
}

// ============================================================
// SiLU (Swish): silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================
inline void silu_inplace(Tensor& x) {
    float*  p = x.data();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) {
        float v = p[i];
        p[i]    = v / (1.0f + std::exp(-v));
    }
}

}  // namespace llamacpp
