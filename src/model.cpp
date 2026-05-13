#include "llamacpp/model.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "llamacpp/dequant.h"

namespace llamacpp {

static Tensor LoadAndDequantize(std::ifstream& in,
                                uint64_t base_offset,
                                const GGUFTensorInfo& tinfo) {
    int64_t n_elements = 1;
    for (int64_t d : tinfo.shape) n_elements *= d;

    size_t n_bytes = BytesPerTensor(tinfo.type, n_elements);
    std::vector<uint8_t> raw(n_bytes);

    in.seekg(static_cast<std::streamoff>(base_offset + tinfo.offset));
    in.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(n_bytes));
    if (!in) {
        throw std::runtime_error("model: short read for tensor " + tinfo.name);
    }

    Tensor t(tinfo.shape);
    DequantizeTensor(tinfo.type, raw.data(), t.data(), n_elements);
    return t;
}

static const GGUFTensorInfo& FindTensor(const std::vector<GGUFTensorInfo>& tensors,
                                        const std::string& name) {
    for (const auto& t : tensors) {
        if (t.name == name) return t;
    }
    throw std::runtime_error("model: tensor not found: " + name);
}

static bool HasTensor(const std::vector<GGUFTensorInfo>& tensors,
                      const std::string& name) {
    for (const auto& t : tensors) {
        if (t.name == name) return true;
    }
    return false;
}

std::unique_ptr<LlamaModel> LlamaModel::Load(const std::string& gguf_path,
                                             bool verbose) {
    auto gguf = GGUFFile::Open(gguf_path);

    LlamaConfig cfg;
    cfg.n_layers         = (int)gguf->GetU64("llama.block_count");
    cfg.hidden_dim       = (int)gguf->GetU64("llama.embedding_length");
    cfg.intermediate_dim = (int)gguf->GetU64("llama.feed_forward_length");
    cfg.n_heads          = (int)gguf->GetU64("llama.attention.head_count");
    cfg.n_kv_heads       = (int)gguf->GetU64("llama.attention.head_count_kv");
    cfg.rope_dim         = (int)gguf->GetU64("llama.rope.dimension_count");
    cfg.head_dim         = cfg.hidden_dim / cfg.n_heads;
    cfg.context_length   = (int)gguf->GetU64("llama.context_length");
    cfg.rms_eps          = gguf->GetF32("llama.attention.layer_norm_rms_epsilon");
    cfg.rope_freq_base   = gguf->HasKey("llama.rope.freq_base")
                                ? gguf->GetF32("llama.rope.freq_base") : 10000.0f;
    cfg.vocab_size       = gguf->HasKey("llama.vocab_size")
                                ? (int)gguf->GetU64("llama.vocab_size") : 0;
    if (gguf->HasKey("tokenizer.ggml.bos_token_id"))
        cfg.bos_token_id = (int)gguf->GetU64("tokenizer.ggml.bos_token_id");
    if (gguf->HasKey("tokenizer.ggml.eos_token_id"))
        cfg.eos_token_id = (int)gguf->GetU64("tokenizer.ggml.eos_token_id");

    if (verbose) {
        std::cout << "[load] Config:" << std::endl;
        std::cout << "  n_layers          = " << cfg.n_layers << std::endl;
        std::cout << "  n_heads           = " << cfg.n_heads << std::endl;
        std::cout << "  n_kv_heads        = " << cfg.n_kv_heads
                  << "  (each KV head shared by "
                  << cfg.n_heads / cfg.n_kv_heads << " Q heads)" << std::endl;
        std::cout << "  hidden_dim        = " << cfg.hidden_dim << std::endl;
        std::cout << "  intermediate_dim  = " << cfg.intermediate_dim << std::endl;
        std::cout << "  head_dim          = " << cfg.head_dim << std::endl;
        std::cout << "  rope_dim          = " << cfg.rope_dim << std::endl;
        std::cout << "  vocab_size        = " << cfg.vocab_size << std::endl;
        std::cout << "  context_length    = " << cfg.context_length << std::endl;
        std::cout << "  rms_eps           = " << cfg.rms_eps << std::endl;
        std::cout << "  rope_freq_base    = " << cfg.rope_freq_base << std::endl;
        std::cout << "  bos_token_id      = " << cfg.bos_token_id << std::endl;
        std::cout << "  eos_token_id      = " << cfg.eos_token_id << std::endl;
    }

    std::ifstream in(gguf_path, std::ios::binary);
    if (!in) throw std::runtime_error("model: cannot reopen " + gguf_path);
    uint64_t base = gguf->TensorDataOffset();

    auto model = std::unique_ptr<LlamaModel>(new LlamaModel());
    model->config = cfg;

    auto t_start = std::chrono::high_resolution_clock::now();

    if (verbose) std::cout << "[load] token_embd.weight ..." << std::endl;
    model->token_embd  = LoadAndDequantize(in, base,
        FindTensor(gguf->Tensors(), "token_embd.weight"));

    if (verbose) std::cout << "[load] output_norm.weight ..." << std::endl;
    model->output_norm = LoadAndDequantize(in, base,
        FindTensor(gguf->Tensors(), "output_norm.weight"));

    if (HasTensor(gguf->Tensors(), "output.weight")) {
        if (verbose) std::cout << "[load] output.weight (untied) ..." << std::endl;
        model->output_w        = LoadAndDequantize(in, base,
            FindTensor(gguf->Tensors(), "output.weight"));
        model->tied_embeddings = false;
    } else {
        if (verbose) std::cout << "[load] output is tied to token_embd" << std::endl;
        model->tied_embeddings = true;
    }

    model->blocks.resize(cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; ++L) {
        if (verbose) {
            std::cout << "[load] block " << L << " / " << (cfg.n_layers - 1) << " ..." << std::endl;
        }
        std::string p = "blk." + std::to_string(L);
        auto& b = model->blocks[L];
        b.attn_norm   = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".attn_norm.weight"));
        b.attn_q      = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".attn_q.weight"));
        b.attn_k      = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".attn_k.weight"));
        b.attn_v      = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".attn_v.weight"));
        b.attn_output = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".attn_output.weight"));
        b.ffn_norm    = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".ffn_norm.weight"));
        b.ffn_gate    = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".ffn_gate.weight"));
        b.ffn_up      = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".ffn_up.weight"));
        b.ffn_down    = LoadAndDequantize(in, base, FindTensor(gguf->Tensors(), p + ".ffn_down.weight"));
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    if (verbose) std::cout << "[load] done in " << ms << " ms" << std::endl;

    return model;
}

}  // namespace llamacpp
