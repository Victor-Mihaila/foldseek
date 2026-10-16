#ifndef MODERNPROST_H
#define MODERNPROST_H

#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <string>
#include <functional>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_gallocr;
struct ggml_backend;
struct ggml_backend_buffer;
struct gguf_context;
typedef struct ggml_gallocr * ggml_gallocr_t;
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

#define MODERNPROST_3DI_ALPHABET  "ACDEFGHIKLMNPQRSTVWY"
#define MODERNPROST_12ST_ALPHABET "ACDEFGHIKLMN"

struct ModernProstHparams {
    uint32_t n_layer = 0, n_embd = 0, n_head = 0, head_dim = 0, n_ff = 0;
    uint32_t sliding_window = 0, noble_rank = 0;
    float rope_global = 0.0f, rope_local = 0.0f, eps = 1e-5f;
    std::vector<int32_t> layer_is_sliding;
};

struct ModernProstNoble {
    ggml_tensor * linear = nullptr;
    ggml_tensor * w_down = nullptr, * w_up = nullptr, * m = nullptr;
    ggml_tensor * omega1 = nullptr, * phi1 = nullptr, * omega2 = nullptr, * phi2 = nullptr;
};

struct ModernProstLayer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * mlp_norm = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
    ModernProstNoble qkv, out;
};

class ModernProstModel {
public:
    ModernProstModel(const std::string & gguf_path, const std::string & device);
    ~ModernProstModel();
    ModernProstModel(const ModernProstModel &) = delete;
    ModernProstModel & operator=(const ModernProstModel &) = delete;

    ModernProstHparams hp;
    bool folded = false;
    std::map<std::string, int32_t> vocab;
    std::vector<ModernProstLayer> layers;
    ggml_tensor * tok_embd = nullptr, * embd_norm = nullptr, * final_norm = nullptr;
    ggml_tensor * head_3di = nullptr, * head_12st = nullptr;

    gguf_context * gguf = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    bool gpu = false;
};

struct ModernProstBatchPolicy {
    int batch_size   = 16;
    int max_residues = 50000;
    int chunk_len    = 4096;
};

class ModernProst {
public:
    ModernProst(ModernProstModel & model, int threads);
    ~ModernProst();

    bool run(const std::string & aa, std::vector<float> & logits_3di,
             std::vector<float> & logits_12st);
    bool predict(const std::string & aa, std::string & three_di,
                 std::string & twelve_st, std::string & packed);

    bool predict_profiles(const std::string & aa,
                          std::vector<uint8_t> & rec_3di,
                          std::vector<uint8_t> & rec_12st);
    bool run_batch(const std::vector<std::string> & seqs,
                   std::vector<std::vector<float>> & logits_3di,
                   std::vector<std::vector<float>> & logits_12st);
    bool predict_profiles_batch(const std::vector<std::string> & seqs,
                                std::vector<std::vector<uint8_t>> & rec_3di,
                                std::vector<std::vector<uint8_t>> & rec_12st);

    bool predict_profiles_many(const std::vector<std::string> & seqs,
                               std::vector<std::vector<uint8_t>> & rec_3di,
                               std::vector<std::vector<uint8_t>> & rec_12st,
                               const ModernProstBatchPolicy & pol = ModernProstBatchPolicy(),
                               const std::function<void(size_t)> & on_done = nullptr);

    void clear_cache();

    ModernProstModel & model;

private:
    struct Cached {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_tensor * inp_tokens = nullptr, * inp_pos = nullptr;
        ggml_tensor * mask_full = nullptr, * mask_slide = nullptr;
        ggml_tensor * out_3di = nullptr, * out_12st = nullptr;
        int n_tokens = 0, n_seq = 1;
        bool mask_f16 = false;
    };
    Cached * get_graph(int n_tokens, int n_seq, int slot = 0);

    struct Staged {
        std::vector<std::vector<int32_t>> ids;
        int n = 0, B = 0;
        Cached * c = nullptr;
        std::vector<int32_t> tok, pos;
        std::vector<uint16_t> m16f, m16s;
        std::vector<float>    m32f, m32s;
    };
    bool stage(const std::vector<std::string> & seqs, Staged & st, int slot);
    bool upload(Staged & st);
    bool compute_async(Staged & st);
    bool collect(Staged & st, std::vector<std::vector<float>> & logits_3di,
                 std::vector<std::vector<float>> & logits_12st);
    ggml_cgraph * build_graph(Cached & c, int n_tokens, int n_seq);
    std::vector<int32_t> tokenize(const std::string & aa) const;
    static int bucket_for(int n);

    int n_threads = 1;
    std::map<std::tuple<int,int,int>, Cached> cache;
};

#endif
