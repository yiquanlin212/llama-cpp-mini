#pragma once

#include <vector>

#include "llamacpp/model.h"
#include "llamacpp/tensor.h"

namespace llamacpp {

struct KVCache {
    int max_seq = 0;
    int kv_dim  = 0;
    std::vector<std::vector<float>> k;
    std::vector<std::vector<float>> v;

    void Init(const LlamaConfig& cfg, int max_seq_len) {
        max_seq = max_seq_len;
        kv_dim  = cfg.n_kv_heads * cfg.head_dim;
        k.assign(cfg.n_layers, std::vector<float>((size_t)max_seq * kv_dim, 0.0f));
        v.assign(cfg.n_layers, std::vector<float>((size_t)max_seq * kv_dim, 0.0f));
    }
};

std::vector<float> Forward(const LlamaModel& model,
                           int token_id,
                           int pos,
                           KVCache& cache);

}  // namespace llamacpp
