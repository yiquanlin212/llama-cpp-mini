#include "llamacpp/forward.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace llamacpp {

// ------------------------------------------------------------
// gemv: y[n] = sum_k x[k] * W[n, k]
// On Apple Silicon: uses NEON FMA (vfmaq_f32) with 4 independent
// accumulators for ILP; unrolled inner loop processes 16 floats
// per iteration. Falls back to scalar on non-ARM.
// ------------------------------------------------------------
static inline void gemv(const float* x, const float* W, float* y,
                        int N, int K) {
#if defined(__ARM_NEON)
    for (int n = 0; n < N; ++n) {
        const float* w_row = W + (size_t)n * K;
        float32x4_t s0 = vdupq_n_f32(0.0f);
        float32x4_t s1 = vdupq_n_f32(0.0f);
        float32x4_t s2 = vdupq_n_f32(0.0f);
        float32x4_t s3 = vdupq_n_f32(0.0f);
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            float32x4_t x0 = vld1q_f32(x + k);
            float32x4_t x1 = vld1q_f32(x + k + 4);
            float32x4_t x2 = vld1q_f32(x + k + 8);
            float32x4_t x3 = vld1q_f32(x + k + 12);
            float32x4_t w0 = vld1q_f32(w_row + k);
            float32x4_t w1 = vld1q_f32(w_row + k + 4);
            float32x4_t w2 = vld1q_f32(w_row + k + 8);
            float32x4_t w3 = vld1q_f32(w_row + k + 12);
            s0 = vfmaq_f32(s0, x0, w0);
            s1 = vfmaq_f32(s1, x1, w1);
            s2 = vfmaq_f32(s2, x2, w2);
            s3 = vfmaq_f32(s3, x3, w3);
        }
        for (; k + 4 <= K; k += 4) {
            float32x4_t xv = vld1q_f32(x + k);
            float32x4_t wv = vld1q_f32(w_row + k);
            s0 = vfmaq_f32(s0, xv, wv);
        }
        float32x4_t s = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        float sum = vaddvq_f32(s);
        for (; k < K; ++k) sum += x[k] * w_row[k];
        y[n] = sum;
    }
#else
    for (int n = 0; n < N; ++n) {
        const float* w_row = W + (size_t)n * K;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += x[k] * w_row[k];
        y[n] = sum;
    }
#endif
}

static inline void rmsnorm(const float* x, const float* w, float* y,
                           int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
    float inv = 1.0f / std::sqrt(sum_sq / (float)dim + eps);
    for (int i = 0; i < dim; ++i) y[i] = x[i] * inv * w[i];
}

static inline void apply_rope(float* x, int pos, int n_heads, int head_dim,
                              int rope_dim, float freq_base) {
    for (int h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim;
        for (int j = 0; j < rope_dim; j += 2) {
            float freq  = std::pow(freq_base, -(float)j / (float)rope_dim);
            float theta = (float)pos * freq;
            float c = std::cos(theta);
            float s = std::sin(theta);
            float a = head[j];
            float b = head[j + 1];
            head[j]     = a * c - b * s;
            head[j + 1] = a * s + b * c;
        }
    }
}

std::vector<float> Forward(const LlamaModel& model,
                           int token_id,
                           int pos,
                           KVCache& cache) {
    const LlamaConfig& cfg = model.config;
    const int H   = cfg.hidden_dim;
    const int FF  = cfg.intermediate_dim;
    const int NH  = cfg.n_heads;
    const int NKH = cfg.n_kv_heads;
    const int HD  = cfg.head_dim;
    const int KVD = NKH * HD;
    const int RD  = cfg.rope_dim;
    const int V   = cfg.vocab_size;

    if (pos >= cache.max_seq) {
        throw std::runtime_error("Forward: position exceeds KV cache size");
    }

    std::vector<float> x(H);
    const float* embd = model.token_embd.data() + (size_t)token_id * H;
    std::memcpy(x.data(), embd, sizeof(float) * H);

    std::vector<float> h_buf(H);
    std::vector<float> q(H);
    std::vector<float> k_t(KVD);
    std::vector<float> v_t(KVD);
    std::vector<float> attn_out(H);
    std::vector<float> proj(H);
    std::vector<float> gate(FF);
    std::vector<float> up(FF);
    std::vector<float> ffn_out(H);
    std::vector<float> scores(pos + 1);

    for (int L = 0; L < cfg.n_layers; ++L) {
        const LlamaBlock& b = model.blocks[L];

        rmsnorm(x.data(), b.attn_norm.data(), h_buf.data(), H, cfg.rms_eps);

        gemv(h_buf.data(), b.attn_q.data(), q.data(),   H,   H);
        gemv(h_buf.data(), b.attn_k.data(), k_t.data(), KVD, H);
        gemv(h_buf.data(), b.attn_v.data(), v_t.data(), KVD, H);

        apply_rope(q.data(),   pos, NH,  HD, RD, cfg.rope_freq_base);
        apply_rope(k_t.data(), pos, NKH, HD, RD, cfg.rope_freq_base);

        std::memcpy(cache.k[L].data() + (size_t)pos * KVD, k_t.data(), sizeof(float) * KVD);
        std::memcpy(cache.v[L].data() + (size_t)pos * KVD, v_t.data(), sizeof(float) * KVD);

        std::memset(attn_out.data(), 0, sizeof(float) * H);
        const int heads_per_kv = NH / NKH;
        const float scale = 1.0f / std::sqrt((float)HD);

        for (int hi = 0; hi < NH; ++hi) {
            int kvh = hi / heads_per_kv;
            const float* qh = q.data() + hi * HD;

            for (int t = 0; t <= pos; ++t) {
                const float* k_pos = cache.k[L].data() + (size_t)t * KVD + kvh * HD;
                float dot = 0.0f;
                for (int d = 0; d < HD; ++d) dot += qh[d] * k_pos[d];
                scores[t] = dot * scale;
            }

            float max_s = scores[0];
            for (int t = 1; t <= pos; ++t) if (scores[t] > max_s) max_s = scores[t];
            float sum = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                sum += scores[t];
            }
            float inv = 1.0f / sum;
            for (int t = 0; t <= pos; ++t) scores[t] *= inv;

            float* out_h = attn_out.data() + hi * HD;
            for (int t = 0; t <= pos; ++t) {
                const float* v_pos = cache.v[L].data() + (size_t)t * KVD + kvh * HD;
                float w = scores[t];
                for (int d = 0; d < HD; ++d) out_h[d] += w * v_pos[d];
            }
        }

        gemv(attn_out.data(), b.attn_output.data(), proj.data(), H, H);
        for (int i = 0; i < H; ++i) x[i] += proj[i];

        rmsnorm(x.data(), b.ffn_norm.data(), h_buf.data(), H, cfg.rms_eps);

        gemv(h_buf.data(), b.ffn_gate.data(), gate.data(), FF, H);
        gemv(h_buf.data(), b.ffn_up.data(),   up.data(),   FF, H);
        for (int i = 0; i < FF; ++i) {
            float g = gate[i];
            float silu = g / (1.0f + std::exp(-g));
            gate[i] = silu * up[i];
        }
        gemv(gate.data(), b.ffn_down.data(), ffn_out.data(), H, FF);
        for (int i = 0; i < H; ++i) x[i] += ffn_out[i];
    }

    rmsnorm(x.data(), model.output_norm.data(), h_buf.data(), H, cfg.rms_eps);

    std::vector<float> logits(V);
    const float* out_w = model.tied_embeddings
                            ? model.token_embd.data()
                            : model.output_w.data();
    gemv(h_buf.data(), out_w, logits.data(), V, H);

    return logits;
}

}  // namespace llamacpp
