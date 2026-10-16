#ifndef PROST_T5_H
#define PROST_T5_H


#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_gallocr;

class ProstT5InitGuard {
public:
    explicit ProstT5InitGuard(bool verbose = false);
    ~ProstT5InitGuard() = default;
    ProstT5InitGuard(const ProstT5InitGuard&) = delete;
    ProstT5InitGuard& operator=(const ProstT5InitGuard&) = delete;
};

struct ProstT5Layer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_up = nullptr, * ffn_down = nullptr;
};

class ProstT5Model {
public:
    ProstT5Model(const std::string& model_file, std::string& device);
    ~ProstT5Model();

    int n_layer = 0, n_embd = 0, n_head = 0, head_dim = 0, n_ff = 0, n_rel_buckets = 0;
    float eps = 1e-6f;
    std::unordered_map<std::string, int> vocab;
    int tok_aa2fold = -1, tok_eos = -1, tok_unk_aa = -1;

    ggml_tensor * tok_embd = nullptr, * rel_b = nullptr, * out_norm = nullptr;
    ggml_tensor * conv0 = nullptr, * conv0_b = nullptr, * conv3 = nullptr, * conv3_b = nullptr;
    std::vector<ProstT5Layer> layers;

    ggml_backend * backend = nullptr;
    bool is_cpu = true;

private:
    gguf_context * gguf = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer * buf_w = nullptr;
    void * map = nullptr;
    size_t map_size = 0;
};

class ProstT5 {
public:
    ProstT5(ProstT5Model& model, int threads);
    ~ProstT5();

    static std::vector<std::string> getDevices();

    std::string predict(const std::string& aa);
    void perf();

    ProstT5Model& model;

private:
    int n_threads = 1;
    ggml_gallocr * galloc = nullptr;
    std::vector<uint8_t> meta;
    size_t n_calls = 0, n_res = 0;
    double t_ms = 0.0;
};

#endif
