#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "llamacpp/gguf.h"

namespace llamacpp {

class Tokenizer {
public:
    explicit Tokenizer(const GGUFFile& gguf);

    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;

    int32_t bos_token_id() const { return bos_token_id_; }
    int32_t eos_token_id() const { return eos_token_id_; }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
    int32_t bos_token_id_ = -1;
    int32_t eos_token_id_ = -1;

    std::vector<std::string> EncodePiece(const std::string& text) const;
    std::string ByteEncode(const std::string& text) const;
    std::string ByteDecode(const std::string& text) const;
    std::string MergeKey(const std::string& a, const std::string& b) const;
};

}  // namespace llamacpp
