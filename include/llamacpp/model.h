#pragma once

#include <memory>
#include <string>
#include <vector>

#include "llamacpp/gguf.h"
#include "llamacpp/tensor.h"

namespace llamacpp {

struct LlamaConfig {
    int   n_layers          = 0;
    int   n_heads           = 0;
    int   n_kv_heads        = 0;
    int   hidden_dim        = 0;
    int   intermediate_dim  = 0;
    int   head_dim          = 0;
    int   rope_dim          = 0;
    int   vocab_size        = 0;
    int   context_length    = 0;
    float rms_eps           = 1e-5f;
    float rope_freq_base    = 10000.0f;
    int   bos_token_id      = -1;
    int   eos_token_id      = -1;
};

struct LlamaBlock {
    Tensor attn_norm;
    Tensor attn_q;
    Tensor attn_k;
    Tensor attn_v;
    Tensor attn_output;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
};

class LlamaModel {
public:
    LlamaConfig             config;
    Tensor                  token_embd;
    Tensor                  output_norm;
    Tensor                  output_w;
    bool                    tied_embeddings = false;
    std::vector<LlamaBlock> blocks;

    static std::unique_ptr<LlamaModel> Load(const std::string& gguf_path,
                                            bool verbose = true);
};

}  // namespace llamacpp
