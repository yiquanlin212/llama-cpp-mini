#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "llamacpp/model.h"
#include "llamacpp/tensor.h"

namespace llamacpp {

struct KVCache {
    int max_seq = 0;
    int kv_dim  = 0;
    int32_t current_pos_ = 0;
    std::vector<std::vector<float>> k;
    std::vector<std::vector<float>> v;

    void Init(const LlamaConfig& cfg, int max_seq_len) {
        max_seq = max_seq_len;
        kv_dim  = cfg.n_kv_heads * cfg.head_dim;
        current_pos_ = 0;
        k.assign(cfg.n_layers, std::vector<float>((size_t)max_seq * kv_dim, 0.0f));
        v.assign(cfg.n_layers, std::vector<float>((size_t)max_seq * kv_dim, 0.0f));
    }

    int32_t AppendPosition() {
        if (current_pos_ < 0 || current_pos_ >= max_seq) {
            throw std::runtime_error("KVCache: position exceeds max_seq");
        }
        return current_pos_++;
    }
};

std::vector<float> Forward(const LlamaModel& model,
                           int token_id,
                           int pos,
                           KVCache& cache);

}  // namespace llamacpp
