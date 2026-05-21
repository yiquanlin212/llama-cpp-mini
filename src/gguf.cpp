#include "llamacpp/gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "llamacpp/dequant.h"

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

class Reader {
public:
    Reader(const uint8_t* data, uint64_t size) : data_(data), size_(size) {}

    template <typename T>
    void ReadPOD(T* out) {
        if (pos_ + sizeof(T) > size_) throw std::runtime_error("GGUF: short read");
        std::memcpy(out, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
    }

    std::string ReadString() {
        uint64_t len;
        ReadPOD(&len);
        if (len > (1ull << 28)) {
            throw std::runtime_error("GGUF: unreasonable string length");
        }
        if (pos_ + len > size_) throw std::runtime_error("GGUF: short read on string");
        std::string s(reinterpret_cast<const char*>(data_ + pos_), static_cast<size_t>(len));
        pos_ += len;
        return s;
    }

    const uint8_t* ReadBytes(uint64_t n) {
        if (pos_ + n > size_) throw std::runtime_error("GGUF: short read");
        const uint8_t* p = data_ + pos_;
        pos_ += n;
        return p;
    }

    uint64_t pos() const { return pos_; }

private:
    const uint8_t* data_;
    uint64_t size_;
    uint64_t pos_ = 0;
};

static void ReadScalar(Reader& r, GGUFType type, GGUFValue* v) {
    v->type = type;
    switch (type) {
        case GGUFType::UINT8:   r.ReadPOD(&v->scalar.u8);  break;
        case GGUFType::INT8:    r.ReadPOD(&v->scalar.i8);  break;
        case GGUFType::UINT16:  r.ReadPOD(&v->scalar.u16); break;
        case GGUFType::INT16:   r.ReadPOD(&v->scalar.i16); break;
        case GGUFType::UINT32:  r.ReadPOD(&v->scalar.u32); break;
        case GGUFType::INT32:   r.ReadPOD(&v->scalar.i32); break;
        case GGUFType::FLOAT32: r.ReadPOD(&v->scalar.f32); break;
        case GGUFType::BOOL: {
            uint8_t b; r.ReadPOD(&b); v->scalar.b = (b != 0); break;
        }
        case GGUFType::STRING:  v->str = r.ReadString(); break;
        case GGUFType::UINT64:  r.ReadPOD(&v->scalar.u64); break;
        case GGUFType::INT64:   r.ReadPOD(&v->scalar.i64); break;
        case GGUFType::FLOAT64: r.ReadPOD(&v->scalar.f64); break;
        default:
            throw std::runtime_error("GGUF: unexpected scalar type");
    }
}

static GGUFValue ReadValue(Reader& r, GGUFType type) {
    GGUFValue v;
    v.type = type;
    if (type == GGUFType::ARRAY) {
        uint32_t elem_type_u;
        uint64_t len;
        r.ReadPOD(&elem_type_u);
        r.ReadPOD(&len);
        v.array_elem_type = static_cast<GGUFType>(elem_type_u);
        v.array.reserve(static_cast<size_t>(len));
        for (uint64_t i = 0; i < len; ++i) {
            GGUFValue elem;
            ReadScalar(r, v.array_elem_type, &elem);
            v.array.push_back(std::move(elem));
        }
    } else {
        ReadScalar(r, type, &v);
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
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("GGUF: cannot open " + path);

    struct stat st;
    if (::fstat(fd, &st) < 0) {
        ::close(fd);
        throw std::runtime_error("GGUF: fstat failed");
    }
    if (st.st_size <= 0) {
        ::close(fd);
        throw std::runtime_error("GGUF: empty file");
    }

    void* mapped = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                          PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == reinterpret_cast<void*>(-1)) {
        ::close(fd);
        throw std::runtime_error("GGUF: mmap failed");
    }

    auto f = std::unique_ptr<GGUFFile>(new GGUFFile());
    f->fd_ = fd;
    f->data_ = static_cast<const uint8_t*>(mapped);
    f->file_size_ = static_cast<uint64_t>(st.st_size);

    Reader r(f->data_, f->file_size_);
    const uint8_t* magic = r.ReadBytes(4);
    if (std::string(reinterpret_cast<const char*>(magic), 4) != "GGUF") {
        throw std::runtime_error("GGUF: bad magic (not a GGUF file)");
    }

    r.ReadPOD(&f->version_);
    if (f->version_ < 2 || f->version_ > 3) {
        throw std::runtime_error("GGUF: unsupported version " + std::to_string(f->version_));
    }
    r.ReadPOD(&f->tensor_count_);
    r.ReadPOD(&f->kv_count_);

    if (f->tensor_count_ > 100000 || f->kv_count_ > 10000) {
        throw std::runtime_error("GGUF: unreasonable counts (corrupt header?)");
    }

    for (uint64_t i = 0; i < f->kv_count_; ++i) {
        std::string key = r.ReadString();
        uint32_t type_u;
        r.ReadPOD(&type_u);
        GGUFValue v = ReadValue(r, static_cast<GGUFType>(type_u));
        f->kv_.emplace(std::move(key), std::move(v));
    }

    if (f->kv_.count("general.alignment")) {
        const auto& v = f->kv_.at("general.alignment");
        if (v.type == GGUFType::UINT32) f->alignment_ = v.scalar.u32;
    }

    f->tensors_.reserve(static_cast<size_t>(f->tensor_count_));
    for (uint64_t i = 0; i < f->tensor_count_; ++i) {
        GGUFTensorInfo t;
        t.name = r.ReadString();
        uint32_t n_dims;
        r.ReadPOD(&n_dims);
        if (n_dims == 0 || n_dims > 4) {
            throw std::runtime_error("GGUF: tensor has unexpected n_dims = " + std::to_string(n_dims));
        }
        // GGUF stores dims in reverse vs math convention; for a [out, in]
        // weight it stores [in, out]. We reverse here so .shape is [out, in].
        std::vector<int64_t> raw_dims(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim;
            r.ReadPOD(&dim);
            raw_dims[d] = static_cast<int64_t>(dim);
        }
        std::reverse(raw_dims.begin(), raw_dims.end());
        t.shape = std::move(raw_dims);

        uint32_t type_u;
        r.ReadPOD(&type_u);
        t.type = static_cast<GGMLType>(type_u);
        r.ReadPOD(&t.offset);

        f->tensors_.push_back(std::move(t));
    }

    uint64_t cur = r.pos();
    uint64_t aligned = ((cur + f->alignment_ - 1) / f->alignment_) * f->alignment_;
    f->tensor_data_offset_ = aligned;

    return f;
}

GGUFFile::~GGUFFile() {
    if (data_) {
        ::munmap(const_cast<uint8_t*>(data_), static_cast<size_t>(file_size_));
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
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

Tensor GGUFFile::load_tensor_f32(const std::string& name) const {
    const GGUFTensorInfo* info = nullptr;
    for (const auto& t : tensors_) {
        if (t.name == name) {
            info = &t;
            break;
        }
    }
    if (!info) throw std::runtime_error("GGUF: tensor not found: " + name);

    int64_t n_elements = 1;
    for (int64_t d : info->shape) n_elements *= d;

    size_t n_bytes = BytesPerTensor(info->type, n_elements);
    uint64_t start = tensor_data_offset_ + info->offset;
    if (start + n_bytes > file_size_) {
        throw std::runtime_error("GGUF: tensor data out of range: " + name);
    }

    Tensor t(info->shape);
    DequantizeTensor(info->type, data_ + start, t.data(), n_elements);
    return t;
}

}  // namespace llamacpp
