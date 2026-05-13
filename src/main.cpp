#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llamacpp/forward.h"
#include "llamacpp/model.h"

using namespace llamacpp;

static void PrintTopK(const std::vector<float>& logits, int k = 5) {
    int n = (int)logits.size();
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return logits[a] > logits[b]; });
    std::cout << "  top-" << k << ":";
    for (int i = 0; i < k; ++i) {
        std::cout << "  [" << idx[i] << " : " << logits[idx[i]] << "]";
    }
    std::cout << std::endl;
}

static int Argmax(const std::vector<float>& logits) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < (int)logits.size(); ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string tokens_arg;
    int n_generate = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens_arg = argv[++i];
        else if (a == "--n-generate" && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else {
            std::cerr << "Unknown arg: " << a << std::endl;
            return 1;
        }
    }
    if (model_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --model <gguf> [--tokens \"id1 id2 ...\"] [--n-generate N]"
                  << std::endl;
        return 1;
    }

    std::cout << "===== Loading model =====" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto model = LlamaModel::Load(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Loaded in " << load_s << " s" << std::endl;

    const LlamaConfig& cfg = model->config;

    std::cout << "\n===== Step 4 sanity check =====" << std::endl;
    std::cout << "token_embd shape: [" << model->token_embd.shape()[0]
              << ", " << model->token_embd.shape()[1] << "]" << std::endl;
    std::cout << "Token 0 (<|begin_of_text|> for Llama 3.2), first 16 dims:" << std::endl;
    std::cout << " ";
    for (int i = 0; i < 16; ++i) {
        std::cout << " " << model->token_embd.data()[i];
    }
    std::cout << std::endl;

    if (tokens_arg.empty()) {
        std::cout << "\nNo --tokens given; load-and-dequant verification complete." << std::endl;
        return 0;
    }

    std::vector<int> prompt;
    {
        std::istringstream iss(tokens_arg);
        int t;
        while (iss >> t) prompt.push_back(t);
    }
    if (prompt.empty()) {
        std::cerr << "--tokens parsed to empty list" << std::endl;
        return 1;
    }

    std::cout << "\n===== Step 5: Forward pass =====" << std::endl;

    KVCache cache;
    int max_seq = std::min(cfg.context_length,
                           std::max(1024, (int)prompt.size() + n_generate + 16));
    cache.Init(cfg, max_seq);

    std::vector<float> logits;
    auto p0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < prompt.size(); ++i) {
        logits = Forward(*model, prompt[i], (int)i, cache);
        std::cout << "[pos=" << i << "  token=" << prompt[i] << "] ";
        PrintTopK(logits, 5);
    }
    auto p1 = std::chrono::high_resolution_clock::now();
    double prefill_s = std::chrono::duration<double>(p1 - p0).count();
    std::cout << "Prefill: " << prompt.size() << " tokens in "
              << prefill_s << " s ("
              << (prompt.size() / prefill_s) << " tok/s)" << std::endl;

    if (n_generate > 0) {
        std::cout << "\n===== Greedy decode =====" << std::endl;
        auto g0 = std::chrono::high_resolution_clock::now();
        int pos = (int)prompt.size();
        std::vector<int> generated;
        for (int step = 0; step < n_generate; ++step) {
            int next = Argmax(logits);
            generated.push_back(next);
            if (next == cfg.eos_token_id) {
                std::cout << "[step " << step << "] EOS at token " << next << std::endl;
                break;
            }
            std::cout << "[step " << step << "  pos=" << pos
                      << "] sampled token " << next << std::endl;
            logits = Forward(*model, next, pos, cache);
            ++pos;
        }
        auto g1 = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(g1 - g0).count();
        std::cout << "Generated " << generated.size() << " tokens in "
                  << gen_s << " s ("
                  << (generated.size() / gen_s) << " tok/s)" << std::endl;

        std::cout << "\nFull token sequence: ";
        for (int t : prompt) std::cout << t << " ";
        std::cout << "|";
        for (int t : generated) std::cout << " " << t;
        std::cout << std::endl;
    }

    return 0;
}
