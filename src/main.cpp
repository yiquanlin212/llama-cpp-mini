#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "llamacpp/forward.h"
#include "llamacpp/model.h"
#include "llamacpp/tokenizer.h"

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

static int SampleToken(const std::vector<float>& logits, float temp) {
    if (temp <= 0.0f) return Argmax(logits);

    static std::mt19937 rng(42);
    float max_v = logits[0];
    for (float v : logits) if (v > max_v) max_v = v;

    std::vector<double> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_v) / temp);
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

static void PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " -m <gguf> [-p <prompt>] [-n N] [--temp T]\n"
              << "       " << argv0
              << " --model <gguf> --tokens \"id1 id2 ...\" [--n-generate N]"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    std::string tokens_arg;
    int n_generate = 64;
    float temp = 0.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) model_path = argv[++i];
        else if ((a == "-p" || a == "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) tokens_arg = argv[++i];
        else if (a == "--n-generate" && i + 1 < argc) n_generate = std::atoi(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) temp = std::atof(argv[++i]);
        else if (a == "--help") {
            PrintUsage(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown arg: " << a << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::cout << "===== Loading model =====" << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto model = LlamaModel::Load(model_path, tokens_arg.empty() ? false : true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Loaded in " << load_s << " s" << std::endl;

    const LlamaConfig& cfg = model->config;

    if (tokens_arg.empty()) {
        if (prompt.empty()) {
            PrintUsage(argv[0]);
            return 1;
        }

        auto gguf = GGUFFile::Open(model_path);
        Tokenizer tok(*gguf);
        std::vector<int32_t> input = tok.encode(prompt, true);

        KVCache cache;
        int max_seq = std::min(cfg.context_length,
                               std::max(1024, (int)input.size() + n_generate + 16));
        cache.Init(cfg, max_seq);

        std::vector<float> logits;
        auto p0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < input.size(); ++i) {
            logits = Forward(*model, input[i], cache.AppendPosition(), cache);
        }
        auto p1 = std::chrono::high_resolution_clock::now();
        double prefill_s = std::chrono::duration<double>(p1 - p0).count();

        std::cout << "\nPrompt: " << prompt << std::endl;
        std::cout << "Output: " << std::flush;

        auto g0 = std::chrono::high_resolution_clock::now();
        int produced = 0;
        for (int step = 0; step < n_generate; ++step) {
            int next = SampleToken(logits, temp);
            if (next == cfg.eos_token_id) break;
            std::cout << tok.decode({next}, true) << std::flush;
            logits = Forward(*model, next, cache.AppendPosition(), cache);
            ++produced;
        }
        auto g1 = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(g1 - g0).count();
        std::cout << std::endl;
        std::cout << "Prefill: " << input.size() << " tokens in "
                  << prefill_s << " s (" << (input.size() / prefill_s)
                  << " tok/s)" << std::endl;
        std::cout << "Decode: " << produced << " tokens in "
                  << gen_s << " s (" << (produced / gen_s)
                  << " tok/s)" << std::endl;
        return 0;
    }

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

    std::vector<int> token_prompt;
    {
        std::istringstream iss(tokens_arg);
        int t;
        while (iss >> t) token_prompt.push_back(t);
    }
    if (token_prompt.empty()) {
        std::cerr << "--tokens parsed to empty list" << std::endl;
        return 1;
    }

    std::cout << "\n===== Step 5: Forward pass =====" << std::endl;

    KVCache cache;
    int max_seq = std::min(cfg.context_length,
                           std::max(1024, (int)token_prompt.size() + n_generate + 16));
    cache.Init(cfg, max_seq);

    std::vector<float> logits;
    auto p0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < token_prompt.size(); ++i) {
        logits = Forward(*model, token_prompt[i], cache.AppendPosition(), cache);
        std::cout << "[pos=" << i << "  token=" << token_prompt[i] << "] ";
        PrintTopK(logits, 5);
    }
    auto p1 = std::chrono::high_resolution_clock::now();
    double prefill_s = std::chrono::duration<double>(p1 - p0).count();
    std::cout << "Prefill: " << token_prompt.size() << " tokens in "
              << prefill_s << " s ("
              << (token_prompt.size() / prefill_s) << " tok/s)" << std::endl;

    if (n_generate > 0) {
        std::cout << "\n===== Greedy decode =====" << std::endl;
        auto g0 = std::chrono::high_resolution_clock::now();
        std::vector<int> generated;
        for (int step = 0; step < n_generate; ++step) {
            int next = Argmax(logits);
            generated.push_back(next);
            if (next == cfg.eos_token_id) {
                std::cout << "[step " << step << "] EOS at token " << next << std::endl;
                break;
            }
            int pos = cache.AppendPosition();
            std::cout << "[step " << step << "  pos=" << pos
                      << "] sampled token " << next << std::endl;
            logits = Forward(*model, next, pos, cache);
        }
        auto g1 = std::chrono::high_resolution_clock::now();
        double gen_s = std::chrono::duration<double>(g1 - g0).count();
        std::cout << "Generated " << generated.size() << " tokens in "
                  << gen_s << " s ("
                  << (generated.size() / gen_s) << " tok/s)" << std::endl;

        std::cout << "\nFull token sequence: ";
        for (int t : token_prompt) std::cout << t << " ";
        std::cout << "|";
        for (int t : generated) std::cout << " " << t;
        std::cout << std::endl;
    }

    return 0;
}
