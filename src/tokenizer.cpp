#include "llamacpp/tokenizer.h"

#include <cctype>
#include <stdexcept>
#include <unordered_set>

namespace llamacpp {

static std::string Utf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

static bool ReadUtf8(const std::string& s, size_t* pos, uint32_t* cp) {
    if (*pos >= s.size()) return false;
    uint8_t c = static_cast<uint8_t>(s[*pos]);
    if (c < 0x80) {
        *cp = c;
        (*pos)++;
        return true;
    }
    int n = 0;
    if ((c & 0xE0) == 0xC0) { *cp = c & 0x1F; n = 2; }
    else if ((c & 0xF0) == 0xE0) { *cp = c & 0x0F; n = 3; }
    else if ((c & 0xF8) == 0xF0) { *cp = c & 0x07; n = 4; }
    else return false;
    if (*pos + n > s.size()) return false;
    for (int i = 1; i < n; ++i) {
        uint8_t x = static_cast<uint8_t>(s[*pos + i]);
        if ((x & 0xC0) != 0x80) return false;
        *cp = (*cp << 6) | (x & 0x3F);
    }
    *pos += n;
    return true;
}

static std::vector<uint8_t> ByteList() {
    std::vector<uint8_t> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(static_cast<uint8_t>(b));
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(static_cast<uint8_t>(b));
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(static_cast<uint8_t>(b));
    return bs;
}

Tokenizer::Tokenizer(const GGUFFile& gguf) {
    auto tok_it = gguf.Metadata().find("tokenizer.ggml.tokens");
    if (tok_it == gguf.Metadata().end() || tok_it->second.type != GGUFType::ARRAY ||
        tok_it->second.array_elem_type != GGUFType::STRING) {
        throw std::runtime_error("Tokenizer: tokenizer.ggml.tokens missing");
    }

    vocab_.reserve(tok_it->second.array.size());
    for (size_t i = 0; i < tok_it->second.array.size(); ++i) {
        vocab_.push_back(tok_it->second.array[i].str);
        token_to_id_[vocab_.back()] = static_cast<int32_t>(i);
    }

    auto merge_it = gguf.Metadata().find("tokenizer.ggml.merges");
    if (merge_it == gguf.Metadata().end() || merge_it->second.type != GGUFType::ARRAY ||
        merge_it->second.array_elem_type != GGUFType::STRING) {
        throw std::runtime_error("Tokenizer: tokenizer.ggml.merges missing");
    }
    merge_rank_.reserve(merge_it->second.array.size() * 2);
    for (size_t i = 0; i < merge_it->second.array.size(); ++i) {
        const std::string& m = merge_it->second.array[i].str;
        size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        merge_rank_[MergeKey(m.substr(0, sp), m.substr(sp + 1))] = static_cast<int32_t>(i);
    }

    if (gguf.HasKey("tokenizer.ggml.bos_token_id"))
        bos_token_id_ = static_cast<int32_t>(gguf.GetU64("tokenizer.ggml.bos_token_id"));
    if (gguf.HasKey("tokenizer.ggml.eos_token_id"))
        eos_token_id_ = static_cast<int32_t>(gguf.GetU64("tokenizer.ggml.eos_token_id"));

    std::vector<uint8_t> bs = ByteList();
    std::unordered_set<int> used(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (used.count(b) == 0) bs.push_back(static_cast<uint8_t>(b));
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        uint32_t cp = i < ByteList().size() ? bs[i] : 256 + n++;
        byte_decoder_[cp] = bs[i];
    }
}

std::string Tokenizer::MergeKey(const std::string& a, const std::string& b) const {
    std::string key = a;
    key.push_back('\0');
    key += b;
    return key;
}

std::string Tokenizer::ByteEncode(const std::string& text) const {
    std::vector<uint8_t> bs = ByteList();
    std::unordered_set<int> used(bs.begin(), bs.end());
    std::unordered_map<int, uint32_t> enc;
    int n = 0;
    for (size_t i = 0; i < bs.size(); ++i) enc[bs[i]] = bs[i];
    for (int b = 0; b < 256; ++b) {
        if (used.count(b) == 0) enc[b] = 256 + n++;
    }

    std::string out;
    for (uint8_t c : text) out += Utf8(enc[c]);
    return out;
}

std::vector<std::string> Tokenizer::EncodePiece(const std::string& text) const {
    std::string enc = ByteEncode(text);
    std::vector<std::string> parts;
    for (size_t i = 0; i < enc.size();) {
        size_t pos = i;
        uint32_t cp = 0;
        if (!ReadUtf8(enc, &pos, &cp)) {
            parts.push_back(enc.substr(i, 1));
            ++i;
        } else {
            parts.push_back(enc.substr(i, pos - i));
            i = pos;
        }
    }

    while (parts.size() > 1) {
        int best_rank = -1;
        size_t best = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            auto it = merge_rank_.find(MergeKey(parts[i], parts[i + 1]));
            if (it != merge_rank_.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank < 0) break;
        parts[best] += parts[best + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    return parts;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_token_id_ >= 0) ids.push_back(bos_token_id_);

    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isspace(c)) {
            while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
            if (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) {
                while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) ++i;
            } else if (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
                int cnt = 0;
                while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])) && cnt < 3) {
                    ++i; ++cnt;
                }
            } else if (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) {
                while (i < text.size() && !std::isalnum(static_cast<unsigned char>(text[i])) &&
                       !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
            }
        } else if (std::isalpha(c)) {
            while (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) ++i;
        } else if (std::isdigit(c)) {
            int cnt = 0;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])) && cnt < 3) {
                ++i; ++cnt;
            }
        } else {
            ++i;
            while (i < text.size() && !std::isalnum(static_cast<unsigned char>(text[i])) &&
                   !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        }

        for (const auto& p : EncodePiece(text.substr(start, i - start))) {
            auto it = token_to_id_.find(p);
            if (it == token_to_id_.end()) throw std::runtime_error("Tokenizer: token not in vocab");
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string Tokenizer::ByteDecode(const std::string& text) const {
    std::string out;
    for (size_t i = 0; i < text.size();) {
        size_t pos = i;
        uint32_t cp = 0;
        if (!ReadUtf8(text, &pos, &cp)) {
            out.push_back(text[i++]);
            continue;
        }
        auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            out += text.substr(i, pos - i);
        }
        i = pos;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string joined;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) continue;
        if (skip_special && (id == bos_token_id_ || id == eos_token_id_)) continue;
        const std::string& t = vocab_[id];
        if (skip_special && t.size() >= 4 && t.rfind("<|", 0) == 0 &&
            t.substr(t.size() - 2) == "|>") continue;
        joined += t;
    }
    return ByteDecode(joined);
}

}  // namespace llamacpp
