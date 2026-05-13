#include "llamacpp/gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace llamacpp {

const char* GGMLTypeName(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0";
        case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::Q8_K: return "Q8_K";
        case GGMLType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

template <typename T>
static void ReadPOD(std::istream& in, T* out) {
    in.read(reinterpret_cast<char*>(out), sizeof(T));
    if (!in) throw std::runtime_error("GGUF: short read");
}

static std::string ReadString(std::istream& in) {
    uint64_t len;
    ReadPOD(in, &len);
    if (len > (1ull << 28)) {
        throw std::runtime_error("GGUF: unreasonable string length");
    }
    std::string s(static_cast<size_t>(len), '\0');
    if (len > 0) {
        in.read(s.data(), static_cast<std::streamsize>(len));
        if (!in) throw std::runtime_error("GGUF: short read on string");
    }
    return s;
}

static void ReadScalar(std::istream& in, GGUFType type, GGUFValue* v) {
    v->type = type;
    switch (type) {
        case GGUFType::UINT8:   ReadPOD(in, &v->scalar.u8);  break;
        case GGUFType::INT8:    ReadPOD(in, &v->scalar.i8);  break;
        case GGUFType::UINT16:  ReadPOD(in, &v->scalar.u16); break;
        case GGUFType::INT16:   ReadPOD(in, &v->scalar.i16); break;
        case GGUFType::UINT32:  ReadPOD(in, &v->scalar.u32); break;
        case GGUFType::INT32:   ReadPOD(in, &v->scalar.i32); break;
        case GGUFType::FLOAT32: ReadPOD(in, &v->scalar.f32); break;
        case GGUFType::BOOL: {
            uint8_t b; ReadPOD(in, &b); v->scalar.b = (b != 0); break;
        }
        case GGUFType::STRING:  v->str = ReadString(in); break;
        case GGUFType::UINT64:  ReadPOD(in, &v->scalar.u64); break;
        case GGUFType::INT64:   ReadPOD(in, &v->scalar.i64); break;
        case GGUFType::FLOAT64: ReadPOD(in, &v->scalar.f64); break;
        default:
            throw std::runtime_error("GGUF: unexpected scalar type");
    }
}

static GGUFValue ReadValue(std::istream& in, GGUFType type) {
    GGUFValue v;
    v.type = type;
    if (type == GGUFType::ARRAY) {
        uint32_t elem_type_u;
        uint64_t len;
        ReadPOD(in, &elem_type_u);
        ReadPOD(in, &len);
        v.array_elem_type = static_cast<GGUFType>(elem_type_u);
        v.array.reserve(static_cast<size_t>(std::min<uint64_t>(len, 1024)));
        for (uint64_t i = 0; i < len; ++i) {
            GGUFValue elem;
            ReadScalar(in, v.array_elem_type, &elem);
            v.array.push_back(std::move(elem));
        }
    } else {
        ReadScalar(in, type, &v);
    }
    return v;
}

std::string GGUFValue::ToString() const {
    std::ostringstream os;
    switch (type) {
        case GGUFType::UINT8:   os << static_cast<int>(scalar.u8); break;
        case GGUFType::INT8:    os << static_cast<int>(scalar.i8); break;
        case GGUFType::UINT16:  os << scalar.u16; break;
        case GGUFType::INT16:   os << scalar.i16; break;
        case GGUFType::UINT32:  os << scalar.u32; break;
        case GGUFType::INT32:   os << scalar.i32; break;
        case GGUFType::FLOAT32: os << scalar.f32; break;
        case GGUFType::BOOL:    os << (scalar.b ? "true" : "false"); break;
        case GGUFType::STRING: {
            std::string s = str;
            if (s.size() > 80) s = s.substr(0, 77) + "...";
            os << "\"" << s << "\"";
            break;
        }
        case GGUFType::UINT64:  os << scalar.u64; break;
        case GGUFType::INT64:   os << scalar.i64; break;
        case GGUFType::FLOAT64: os << scalar.f64; break;
        case GGUFType::ARRAY:
            os << "[array of " << array.size() << "]";
            break;
        default: os << "?"; break;
    }
    return os.str();
}

std::unique_ptr<GGUFFile> GGUFFile::Open(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("GGUF: cannot open " + path);

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "GGUF") {
        throw std::runtime_error("GGUF: bad magic (not a GGUF file)");
    }

    auto f = std::unique_ptr<GGUFFile>(new GGUFFile());
    ReadPOD(in, &f->version_);
    if (f->version_ < 2 || f->version_ > 3) {
        throw std::runtime_error("GGUF: unsupported version " + std::to_string(f->version_));
    }
    ReadPOD(in, &f->tensor_count_);
    ReadPOD(in, &f->kv_count_);

    if (f->tensor_count_ > 100000 || f->kv_count_ > 10000) {
        throw std::runtime_error("GGUF: unreasonable counts (corrupt header?)");
    }

    for (uint64_t i = 0; i < f->kv_count_; ++i) {
        std::string key = ReadString(in);
        uint32_t type_u;
        ReadPOD(in, &type_u);
        GGUFValue v = ReadValue(in, static_cast<GGUFType>(type_u));
        f->kv_.emplace(std::move(key), std::move(v));
    }

    if (f->kv_.count("general.alignment")) {
        const auto& v = f->kv_.at("general.alignment");
        if (v.type == GGUFType::UINT32) f->alignment_ = v.scalar.u32;
    }

    f->tensors_.reserve(static_cast<size_t>(f->tensor_count_));
    for (uint64_t i = 0; i < f->tensor_count_; ++i) {
        GGUFTensorInfo t;
        t.name = ReadString(in);
        uint32_t n_dims;
        ReadPOD(in, &n_dims);
        if (n_dims == 0 || n_dims > 4) {
            throw std::runtime_error("GGUF: tensor has unexpected n_dims = " + std::to_string(n_dims));
        }
        // GGUF stores dims in reverse vs math convention; for a [out, in]
        // weight it stores [in, out]. We reverse here so .shape is [out, in].
        std::vector<int64_t> raw_dims(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim;
            ReadPOD(in, &dim);
            raw_dims[d] = static_cast<int64_t>(dim);
        }
        std::reverse(raw_dims.begin(), raw_dims.end());
        t.shape = std::move(raw_dims);

        uint32_t type_u;
        ReadPOD(in, &type_u);
        t.type = static_cast<GGMLType>(type_u);
        ReadPOD(in, &t.offset);

        f->tensors_.push_back(std::move(t));
    }

    uint64_t cur = static_cast<uint64_t>(in.tellg());
    uint64_t aligned = ((cur + f->alignment_ - 1) / f->alignment_) * f->alignment_;
    f->tensor_data_offset_ = aligned;

    return f;
}

uint64_t GGUFFile::GetU64(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("GGUF: missing key " + key);
    const auto& v = it->second;
    switch (v.type) {
        case GGUFType::UINT8:   return v.scalar.u8;
        case GGUFType::UINT16:  return v.scalar.u16;
        case GGUFType::UINT32:  return v.scalar.u32;
        case GGUFType::UINT64:  return v.scalar.u64;
        case GGUFType::INT8:    return static_cast<uint64_t>(v.scalar.i8);
        case GGUFType::INT16:   return static_cast<uint64_t>(v.scalar.i16);
        case GGUFType::INT32:   return static_cast<uint64_t>(v.scalar.i32);
        case GGUFType::INT64:   return static_cast<uint64_t>(v.scalar.i64);
        default:
            throw std::runtime_error("GGUF: key " + key + " is not numeric");
    }
}

float GGUFFile::GetF32(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("GGUF: missing key " + key);
    const auto& v = it->second;
    if (v.type == GGUFType::FLOAT32) return v.scalar.f32;
    if (v.type == GGUFType::FLOAT64) return static_cast<float>(v.scalar.f64);
    throw std::runtime_error("GGUF: key " + key + " is not float");
}

std::string GGUFFile::GetString(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("GGUF: missing key " + key);
    if (it->second.type != GGUFType::STRING) {
        throw std::runtime_error("GGUF: key " + key + " is not string");
    }
    return it->second.str;
}

}  // namespace llamacpp
