#include "llamacpp/dequant.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace llamacpp {

constexpr int    QK_K             = 256;
constexpr int    QK8_0            = 32;
constexpr size_t Q8_0_BLOCK_BYTES = 2 + 32;            // 34 bytes
constexpr size_t Q4_K_BLOCK_BYTES = 2  + 2  + 12 + 128;  // 144 bytes
constexpr size_t Q6_K_BLOCK_BYTES = 128 + 64 + 16 + 2;   // 210 bytes

static inline void get_scale_min_k4(int j, const uint8_t* q,
                                    uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j]     & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

void DequantizeQ8_0(const uint8_t* src, float* dst, int64_t n_elements) {
    if (n_elements % QK8_0 != 0) {
        throw std::runtime_error("DequantizeQ8_0: n_elements must be a multiple of 32");
    }
    int64_t nb = n_elements / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* block = src + i * Q8_0_BLOCK_BYTES;
        uint16_t d_raw;
        std::memcpy(&d_raw, block, 2);
        const float d = fp16_to_fp32(d_raw);
        const int8_t* q = reinterpret_cast<const int8_t*>(block + 2);
        for (int j = 0; j < QK8_0; ++j) {
            dst[i * QK8_0 + j] = d * static_cast<float>(q[j]);
        }
    }
}

void DequantizeQ4_K(const uint8_t* src, float* dst, int64_t n_elements) {
    if (n_elements % QK_K != 0) {
        throw std::runtime_error("DequantizeQ4_K: n_elements must be a multiple of 256");
    }
    int64_t nb = n_elements / QK_K;
    float* y = dst;

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* block = src + i * Q4_K_BLOCK_BYTES;

        uint16_t d_raw, dmin_raw;
        std::memcpy(&d_raw,    block + 0, 2);
        std::memcpy(&dmin_raw, block + 2, 2);
        const float d   = fp16_to_fp32(d_raw);
        const float min = fp16_to_fp32(dmin_raw);

        const uint8_t* scales = block + 4;
        const uint8_t* q      = block + 4 + 12;

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >>   4) - m2;
            q  += 32;
            is += 2;
        }
    }
}

void DequantizeQ6_K(const uint8_t* src, float* dst, int64_t n_elements) {
    if (n_elements % QK_K != 0) {
        throw std::runtime_error("DequantizeQ6_K: n_elements must be a multiple of 256");
    }
    int64_t nb = n_elements / QK_K;
    float* y = dst;

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* block = src + i * Q6_K_BLOCK_BYTES;
        const uint8_t* ql_base = block;
        const uint8_t* qh_base = block + 128;
        const int8_t*  sc_base = reinterpret_cast<const int8_t*>(block + 128 + 64);
        uint16_t d_raw;
        std::memcpy(&d_raw, block + 128 + 64 + 16, 2);
        const float d = fp16_to_fp32(d_raw);

        const uint8_t* ql = ql_base;
        const uint8_t* qh = qh_base;
        const int8_t*  sc = sc_base;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void DequantizeF16(const uint8_t* src, float* dst, int64_t n_elements) {
    for (int64_t i = 0; i < n_elements; ++i) {
        uint16_t h;
        std::memcpy(&h, src + i * 2, 2);
        dst[i] = fp16_to_fp32(h);
    }
}

void DequantizeF32(const uint8_t* src, float* dst, int64_t n_elements) {
    std::memcpy(dst, src, static_cast<size_t>(n_elements) * sizeof(float));
}

void DequantizeTensor(GGMLType type, const uint8_t* src,
                      float* dst, int64_t n_elements) {
    switch (type) {
        case GGMLType::F32:  DequantizeF32 (src, dst, n_elements); break;
        case GGMLType::F16:  DequantizeF16 (src, dst, n_elements); break;
        case GGMLType::Q8_0: DequantizeQ8_0(src, dst, n_elements); break;
        case GGMLType::Q4_K: DequantizeQ4_K(src, dst, n_elements); break;
        case GGMLType::Q6_K: DequantizeQ6_K(src, dst, n_elements); break;
        default:
            throw std::runtime_error(std::string("DequantizeTensor: unsupported type ")
                                     + GGMLTypeName(type));
    }
}

size_t BytesPerTensor(GGMLType type, int64_t n_elements) {
    switch (type) {
        case GGMLType::F32:  return static_cast<size_t>(n_elements) * 4;
        case GGMLType::F16:  return static_cast<size_t>(n_elements) * 2;
        case GGMLType::Q8_0: return static_cast<size_t>(n_elements / QK8_0) * Q8_0_BLOCK_BYTES;
        case GGMLType::Q4_K: return static_cast<size_t>(n_elements / QK_K) * Q4_K_BLOCK_BYTES;
        case GGMLType::Q6_K: return static_cast<size_t>(n_elements / QK_K) * Q6_K_BLOCK_BYTES;
        default:
            throw std::runtime_error(std::string("BytesPerTensor: unsupported type ")
                                     + GGMLTypeName(type));
    }
}

}  // namespace llamacpp
