#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "llamacpp/gguf.h"

namespace llamacpp {

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            int32_t e = -14;
            while ((mant & 0x0400) == 0) { mant <<= 1; --e; }
            mant &= 0x03FF;
            out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mant << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

void DequantizeF32 (const uint8_t* src, float* dst, int64_t n_elements);
void DequantizeF16 (const uint8_t* src, float* dst, int64_t n_elements);
void DequantizeQ8_0(const uint8_t* src, float* dst, int64_t n_elements);
void DequantizeQ4_K(const uint8_t* src, float* dst, int64_t n_elements);
void DequantizeQ6_K(const uint8_t* src, float* dst, int64_t n_elements);

void DequantizeTensor(GGMLType type, const uint8_t* src,
                      float* dst, int64_t n_elements);

size_t BytesPerTensor(GGMLType type, int64_t n_elements);

}  // namespace llamacpp
