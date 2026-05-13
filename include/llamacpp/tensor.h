#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace llamacpp {

// A simple float32, row-major, contiguous N-D tensor.
// Supports 1D to 4D — enough for transformer activations:
//   1D: bias                   [hidden]
//   2D: weights, embeddings    [out, in]
//   3D: activations            [batch, seq, hidden]
//   4D: attention shaped       [batch, heads, seq, head_dim]
class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
        ComputeStrides();
        data_.resize(static_cast<size_t>(numel()), 0.0f);
    }

    // ----- Factories -----
    static Tensor zeros(std::vector<int64_t> shape) {
        return Tensor(std::move(shape));
    }

    static Tensor ones(std::vector<int64_t> shape) {
        Tensor t(std::move(shape));
        t.fill(1.0f);
        return t;
    }

    static Tensor from_values(std::vector<int64_t> shape,
                              std::vector<float> values) {
        Tensor t(std::move(shape));
        if (static_cast<size_t>(t.numel()) != values.size()) {
            throw std::runtime_error("from_values: numel mismatch");
        }
        std::copy(values.begin(), values.end(), t.data_.begin());
        return t;
    }

    // ----- Shape / metadata -----
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape_) n *= d;
        return n;
    }
    const std::vector<int64_t>& shape()   const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }

    // ----- Raw data pointer -----
    float*       data()       { return data_.data(); }
    const float* data() const { return data_.data(); }

    // ----- Linear access (for kernel inner loops) -----
    float& operator[](int64_t i)       { return data_[static_cast<size_t>(i)]; }
    float  operator[](int64_t i) const { return data_[static_cast<size_t>(i)]; }

    // ----- Multi-dim element access -----
    float& at(int64_t i0) {
        assert(ndim() == 1);
        return data_[static_cast<size_t>(i0)];
    }
    float& at(int64_t i0, int64_t i1) {
        assert(ndim() == 2);
        return data_[static_cast<size_t>(i0 * strides_[0] + i1)];
    }
    float& at(int64_t i0, int64_t i1, int64_t i2) {
        assert(ndim() == 3);
        return data_[static_cast<size_t>(
            i0 * strides_[0] + i1 * strides_[1] + i2)];
    }
    float& at(int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
        assert(ndim() == 4);
        return data_[static_cast<size_t>(
            i0 * strides_[0] + i1 * strides_[1]
            + i2 * strides_[2] + i3)];
    }

    // ----- Mutators -----
    void fill(float v) {
        std::fill(data_.begin(), data_.end(), v);
    }

    void random_uniform(float lo, float hi, uint64_t seed = 42) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& v : data_) v = dist(rng);
    }

    // Reshape: must preserve total element count
    void reshape(std::vector<int64_t> new_shape) {
        int64_t new_numel = 1;
        for (int64_t d : new_shape) new_numel *= d;
        if (new_numel != numel()) {
            throw std::runtime_error("reshape: numel mismatch");
        }
        shape_ = std::move(new_shape);
        ComputeStrides();
    }

    // ----- Debug print -----
    void print(const std::string& label = "") const {
        if (!label.empty()) std::cout << label << " ";
        std::cout << "Tensor<shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i + 1 < shape_.size()) std::cout << ",";
        }
        std::cout << "]>" << std::endl;
        PrintData();
    }

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::vector<float>   data_;

    void ComputeStrides() {
        strides_.assign(shape_.size(), 1);
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    void PrintData() const {
        if (ndim() == 1) {
            std::cout << "  [";
            for (int64_t i = 0; i < shape_[0]; ++i) {
                std::cout << std::fixed << std::setprecision(4) << data_[i];
                if (i + 1 < shape_[0]) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else if (ndim() == 2) {
            for (int64_t i = 0; i < shape_[0]; ++i) {
                std::cout << "  [";
                for (int64_t j = 0; j < shape_[1]; ++j) {
                    std::cout << std::fixed << std::setprecision(4)
                              << data_[i * strides_[0] + j];
                    if (j + 1 < shape_[1]) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        } else {
            // For 3D+, print flat with truncation
            int64_t n = numel();
            int64_t show = std::min<int64_t>(n, 16);
            std::cout << "  flat[";
            for (int64_t i = 0; i < show; ++i) {
                std::cout << std::fixed << std::setprecision(4) << data_[i];
                if (i + 1 < show) std::cout << ", ";
            }
            if (n > show) std::cout << ", ... (" << (n - show) << " more)";
            std::cout << "]" << std::endl;
        }
    }
};

}  // namespace llamacpp
