#include "ProstT5.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const char * k3Di = "ACDEFGHIKLMNPQRSTVWY";

static bool g_verbose = false;
static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

ProstT5InitGuard::ProstT5InitGuard(bool verbose) {
    g_verbose = verbose;
    ggml_log_set(quiet_log, nullptr);
}

static int32_t relative_position_bucket(int32_t x, int32_t y, uint64_t n_buckets, bool bidirectional) {
    const int64_t max_distance = 128;
    if (bidirectional) {
        n_buckets >>= 1;
    }
    const int64_t max_exact = n_buckets >> 1;
    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;
    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }
    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);
    return relative_bucket;
}

static ggml_tensor * need(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (t == nullptr) throw std::runtime_error("ProstT5: missing tensor " + name);
    return t;
}

ProstT5Model::ProstT5Model(const std::string& model_file, std::string& device) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = {  true,  &meta };
    gguf = gguf_init_from_file(model_file.c_str(), gp);
    if (gguf == nullptr) throw std::runtime_error("ProstT5: cannot read gguf: " + model_file);

    auto u32 = [&](const char * k) {
        const int64_t i = gguf_find_key(gguf, k);
        if (i < 0) throw std::runtime_error(std::string("ProstT5: missing key ") + k);
        return (int) gguf_get_val_u32(gguf, i);
    };
    auto f32 = [&](const char * k) {
        const int64_t i = gguf_find_key(gguf, k);
        if (i < 0) throw std::runtime_error(std::string("ProstT5: missing key ") + k);
        return gguf_get_val_f32(gguf, i);
    };
    n_layer       = u32("t5encoder.block_count");
    n_embd        = u32("t5encoder.embedding_length");
    n_head        = u32("t5encoder.attention.head_count");
    head_dim      = u32("t5encoder.attention.key_length");
    n_ff          = u32("t5encoder.feed_forward_length");
    n_rel_buckets = u32("t5encoder.attention.relative_buckets_count");
    eps           = f32("t5encoder.attention.layer_norm_rms_epsilon");
    {
        const int64_t i = gguf_find_key(gguf, "tokenizer.ggml.tokens");
        if (i < 0) throw std::runtime_error("ProstT5: missing tokenizer.ggml.tokens");
        for (size_t j = 0; j < gguf_get_arr_n(gguf, i); ++j) vocab[gguf_get_arr_str(gguf, i, j)] = (int) j;
        auto id = [&](const char * s) {
            auto it = vocab.find(s);
            if (it == vocab.end()) throw std::runtime_error(std::string("ProstT5: token not in vocab: ") + s);
            return it->second;
        };
        tok_aa2fold = id("<AA2fold>");
        tok_eos     = id("</s>");
        tok_unk_aa  = id("\xE2\x96\x81X");
    }

    ggml_backend_dev_t dev = nullptr;
    if (!(device.empty() || device == "none" || device == "CPU")) {
        dev = ggml_backend_dev_by_name(device.c_str());
        if (dev == nullptr) throw std::runtime_error("ProstT5: no ggml device named " + device);
    }
    if (dev == nullptr) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) throw std::runtime_error("ProstT5: cannot init backend for " + device);
    is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;

    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 1), nullptr, true };
    ctx_w = ggml_init(ip);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        if (strcmp(name, "output.weight") == 0) continue;
        ggml_tensor * src = ggml_get_tensor(meta, name);
        ggml_tensor * dst = ggml_new_tensor(ctx_w, src->type, GGML_MAX_DIMS, src->ne);
        ggml_set_name(dst, name);
    }

    const size_t data_off = gguf_get_data_offset(gguf);
    int fd = open(model_file.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("ProstT5: cannot open " + model_file);
    struct stat st;
    fstat(fd, &st);
    map_size = (size_t) st.st_size;
    map = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { map = nullptr; throw std::runtime_error("ProstT5: mmap failed for " + model_file); }

    if (is_cpu) {
        buf_w = ggml_backend_cpu_buffer_from_ptr((char *) map + data_off, map_size - data_off);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t; t = ggml_get_next_tensor(ctx_w, t)) {
            const int64_t i = gguf_find_tensor(gguf, ggml_get_name(t));
            ggml_backend_tensor_alloc(buf_w, t, (char *) map + data_off + gguf_get_tensor_offset(gguf, i));
        }
    } else {
        munmap(map, map_size); map = nullptr;
        buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
        if (buf_w == nullptr) throw std::runtime_error("ProstT5: cannot allocate weights on " + device);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        const size_t chunk = 64u << 20;
        ggml_backend_buffer_t stage = host_buft ? ggml_backend_buft_alloc_buffer(host_buft, chunk) : nullptr;
        std::vector<char> fallback;
        char * sbuf = stage ? (char *) ggml_backend_buffer_get_base(stage) : (fallback.resize(chunk), fallback.data());
        int fdr = open(model_file.c_str(), O_RDONLY);
        if (fdr < 0) throw std::runtime_error("ProstT5: cannot open " + model_file);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t; t = ggml_get_next_tensor(ctx_w, t)) {
            const int64_t i = gguf_find_tensor(gguf, ggml_get_name(t));
            const size_t base = data_off + gguf_get_tensor_offset(gguf, i), nb = ggml_nbytes(t);
            for (size_t done = 0; done < nb; ) {
                const size_t n = std::min(chunk, nb - done);
                if (pread(fdr, sbuf, n, (off_t) (base + done)) != (ssize_t) n) {
                    close(fdr); throw std::runtime_error(std::string("ProstT5: short read on ") + ggml_get_name(t));
                }
                ggml_backend_tensor_set(t, sbuf, done, n);
                done += n;
            }
        }
        close(fdr);
        if (stage) ggml_backend_buffer_free(stage);
    }
    ggml_free(meta);

    tok_embd = need(ctx_w, "token_embd.weight");
    rel_b    = need(ctx_w, "enc.blk.0.attn_rel_b.weight");
    out_norm = need(ctx_w, "enc.output_norm.weight");
    conv0    = need(ctx_w, "classifier.0.weight");
    conv0_b  = need(ctx_w, "classifier.0.bias");
    conv3    = need(ctx_w, "classifier.3.weight");
    conv3_b  = need(ctx_w, "classifier.3.bias");
    layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const std::string b = "enc.blk." + std::to_string(il) + ".";
        ProstT5Layer & L = layers[il];
        L.attn_norm = need(ctx_w, b + "attn_norm.weight");
        L.wq        = need(ctx_w, b + "attn_q.weight");
        L.wk        = need(ctx_w, b + "attn_k.weight");
        L.wv        = need(ctx_w, b + "attn_v.weight");
        L.wo        = need(ctx_w, b + "attn_o.weight");
        L.ffn_norm  = need(ctx_w, b + "ffn_norm.weight");
        L.ffn_up    = need(ctx_w, b + "ffn_up.weight");
        L.ffn_down  = need(ctx_w, b + "ffn_down.weight");
    }
}

ProstT5Model::~ProstT5Model() {
    if (buf_w)   ggml_backend_buffer_free(buf_w);
    if (ctx_w)   ggml_free(ctx_w);
    if (backend) ggml_backend_free(backend);
    if (map)     munmap(map, map_size);
    if (gguf)    gguf_free(gguf);
}

ProstT5::ProstT5(ProstT5Model& model, int threads) : model(model), n_threads(std::max(1, threads)) {
    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    meta.resize(ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false));
}

ProstT5::~ProstT5() {
    if (galloc) ggml_gallocr_free(galloc);
}

std::string ProstT5::predict(const std::string& aa) {
    const auto t0 = std::chrono::steady_clock::now();
    const int N = (int) aa.size();
    if (N == 0) return std::string();

    std::vector<int32_t> tok;
    tok.reserve(N + 2);
    tok.push_back(model.tok_aa2fold);
    std::string piece = "\xE2\x96\x81?";
    for (int i = 0; i < N; ++i) {
        piece.back() = (char) toupper((unsigned char) aa[i]);
        auto it = model.vocab.find(piece);
        tok.push_back(it == model.vocab.end() ? model.tok_unk_aa : it->second);
    }
    tok.push_back(model.tok_eos);
    const int T = (int) tok.size();

    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    ggml_tensor * inp_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(inp_tok);
    ggml_tensor * inp_bkt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T, T);
    ggml_set_input(inp_bkt);

    const int H = model.n_head, D = model.head_dim, n_inner = H * D;

    ggml_tensor * pb = ggml_get_rows(ctx, model.rel_b, ggml_view_1d(ctx, inp_bkt, (int64_t) T * T, 0));
    pb = ggml_view_3d(ctx, pb, pb->ne[0], T, T, ggml_element_size(pb) * pb->ne[0],
                      ggml_element_size(pb) * pb->ne[0] * T, 0);
    pb = ggml_cont(ctx, ggml_permute(ctx, pb, 2, 0, 1, 3));

    ggml_tensor * x = ggml_get_rows(ctx, model.tok_embd, inp_tok);
    for (int il = 0; il < model.n_layer; ++il) {
        const ProstT5Layer & L = model.layers[il];
        ggml_tensor * inpSA = x;
        ggml_tensor * cur = ggml_mul(ctx, ggml_rms_norm(ctx, x, model.eps), L.attn_norm);

        ggml_tensor * Q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq, cur), D, H, T);
        ggml_tensor * K = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, cur), D, H, T);
        ggml_tensor * V = ggml_mul_mat(ctx, L.wv, cur);
        ggml_tensor * q = ggml_permute(ctx, Q, 0, 2, 1, 3);
        ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_add(ctx, kq, pb);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * v = ggml_cont(ctx, ggml_transpose(ctx, V));
        ggml_tensor * kqv = ggml_mul_mat(ctx, ggml_reshape_3d(ctx, v, T, D, H), kq);
        cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), n_inner, T);
        cur = ggml_mul_mat(ctx, L.wo, cur);

        ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);
        cur = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, model.eps), L.ffn_norm);
        cur = ggml_mul_mat(ctx, L.ffn_down, ggml_relu(ctx, ggml_mul_mat(ctx, L.ffn_up, cur)));
        x = ggml_add(ctx, cur, ffn_inp);
    }
    x = ggml_mul(ctx, ggml_rms_norm(ctx, x, model.eps), model.out_norm);

    ggml_tensor * h = ggml_view_2d(ctx, x, model.n_embd, N, x->nb[1], x->nb[1]);
    h = ggml_pad(ctx, h, 0, 1, 0, 0);
    h = ggml_cont(ctx, ggml_permute(ctx, h, 2, 0, 1, 3));
    h = ggml_conv_2d(ctx, model.conv0, h, 1, 1, 3, 0, 1, 1);
    h = ggml_add(ctx, h, ggml_reshape_4d(ctx, model.conv0_b, 1, 1, model.conv0_b->ne[0], 1));
    h = ggml_relu(ctx, h);
    h = ggml_conv_2d(ctx, model.conv3, h, 1, 1, 3, 0, 1, 1);
    h = ggml_add(ctx, h, ggml_reshape_4d(ctx, model.conv3_b, 1, 1, model.conv3_b->ne[0], 1));
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    std::string result;
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { ggml_free(ctx); return result; }
    ggml_backend_tensor_set(inp_tok, tok.data(), 0, tok.size() * sizeof(int32_t));
    {
        std::vector<int32_t> bkt((size_t) T * T);
        for (int j = 0; j < T; ++j)
            for (int i = 0; i < T; ++i)
                bkt[(size_t) j * T + i] = relative_position_bucket(i, j, model.n_rel_buckets, true);
        ggml_backend_tensor_set(inp_bkt, bkt.data(), 0, bkt.size() * sizeof(int32_t));
    }
    if (model.is_cpu) ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return result; }

    const int W = N + 1, C = (int) h->ne[2];
    std::vector<float> out((size_t) W * C);
    ggml_backend_tensor_get(h, out.data(), 0, out.size() * sizeof(float));
    ggml_free(ctx);

    result.reserve(N);
    for (int j = 0; j < N; ++j) {
        int best = 0; float bv = std::numeric_limits<float>::lowest();
        for (int c = 0; c < C; ++c) {
            const float v = out[(size_t) c * W + j];
            if (v > bv) { bv = v; best = c; }
        }
        result.push_back(best < 20 ? k3Di[best] : 'X');
    }
    n_calls++; n_res += N;
    t_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

std::vector<std::string> ProstT5::getDevices() {
    std::vector<std::string> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        std::string description = ggml_backend_dev_description(dev);
        // ignore Metal in CI
        if (name == "Metal" && description.find("Paravirtual") != std::string::npos) {
            continue;
        }
        devices.push_back(name);
    }
    return devices;
}

void ProstT5::perf() {
    fprintf(stderr, "ProstT5: %zu sequences, %zu residues, %.1f ms total, %.0f residues/s\n",
            n_calls, n_res, t_ms, t_ms > 0 ? 1000.0 * n_res / t_ms : 0.0);
}
