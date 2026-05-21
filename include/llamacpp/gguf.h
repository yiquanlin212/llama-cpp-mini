#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llamacpp/tensor.h"

namespace llamacpp {

// GGUF metadata value types — match the spec exactly.
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
enum class GGUFType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// GGML tensor data types (subset most Llama GGUFs use).
enum class GGMLType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
};

const char* GGMLTypeName(GGMLType t);

// A single metadata value — supports scalars, strings, and 1-level arrays.
struct GGUFValue {
    GGUFType type = GGUFType::UINT32;

    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        bool     b;
        uint64_t u64;
        int64_t  i64;
        double   f64;
    } scalar = {};

    std::string str;

    GGUFType array_elem_type = GGUFType::UINT32;
    std::vector<GGUFValue> array;

    std::string ToString() const;
};

struct GGUFTensorInfo {
    std::string          name;
    std::vector<int64_t> shape;       // Math-order [outer, ..., inner]
    GGMLType             type   = GGMLType::F32;
    uint64_t             offset = 0;  // byte offset from start of tensor data region
};

class GGUFFile {
public:
    static std::unique_ptr<GGUFFile> Open(const std::string& path);
    ~GGUFFile();

    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    uint32_t Version()       const { return version_; }
    uint64_t TensorCount()   const { return tensor_count_; }
    uint64_t MetadataCount() const { return kv_count_; }

    const std::map<std::string, GGUFValue>& Metadata() const { return kv_; }
    const std::vector<GGUFTensorInfo>&      Tensors()  const { return tensors_; }

    uint64_t TensorDataOffset() const { return tensor_data_offset_; }

    bool        HasKey(const std::string& key)    const { return kv_.count(key) > 0; }
    uint64_t    GetU64(const std::string& key)    const;
    float       GetF32(const std::string& key)    const;
    std::string GetString(const std::string& key) const;
    Tensor      load_tensor_f32(const std::string& name) const;

private:
    GGUFFile() = default;

    uint32_t version_           = 0;
    uint64_t tensor_count_      = 0;
    uint64_t kv_count_          = 0;
    uint64_t tensor_data_offset_= 0;
    uint64_t alignment_         = 32;
    std::map<std::string, GGUFValue> kv_;
    std::vector<GGUFTensorInfo>      tensors_;
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    uint64_t file_size_ = 0;
};

}  // namespace llamacpp
