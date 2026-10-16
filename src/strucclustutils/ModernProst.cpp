#include "ModernProst.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <chrono>

namespace {

ggml_tensor * need(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (t == nullptr) {
        throw std::runtime_error("ModernProst: missing tensor '" + name + "'");
    }
    return t;
}

static bool mp_prec32() {
    static const bool v = (getenv("MP_PREC32") != nullptr);
    return v;
}
static ggml_tensor * mp_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, bool f16_out = false) {
    ggml_tensor * r = ggml_mul_mat(ctx, a, b);
    if (f16_out) {
        r->type  = GGML_TYPE_F16;
        r->nb[0] = ggml_type_size(GGML_TYPE_F16);
        r->nb[1] = r->nb[0] * r->ne[0];
        r->nb[2] = r->nb[1] * r->ne[1];
        r->nb[3] = r->nb[2] * r->ne[2];
    } else if (mp_prec32()) {
        ggml_prec_set_acc(r, GGML_PREC_F32);
    }
    return r;
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    ggml_tensor * n = ggml_norm(ctx, x, eps);
    return w ? ggml_mul(ctx, n, w) : n;
}

ggml_tensor * noble(ggml_context * ctx, ggml_tensor * x, const ModernProstNoble & n, bool f16 = false,
                    bool folded = false, ggml_tensor * residual = nullptr) {
    ggml_tensor * base = mp_mul_mat(ctx, n.linear, x, f16);
    if (n.w_down == nullptr) {
        return residual ? ggml_add(ctx, base, residual) : base;
    }
    ggml_tensor * h = mp_mul_mat(ctx, n.w_down, x, f16);
    h = ggml_cos(ctx, ggml_add(ctx, folded ? h : ggml_mul(ctx, h, n.omega1), n.phi1));
    h = mp_mul_mat(ctx, n.m, h, f16);
    h = ggml_cos(ctx, ggml_add(ctx, folded ? h : ggml_mul(ctx, h, n.omega2), n.phi2));
    ggml_tensor * y = ggml_add(ctx, base, mp_mul_mat(ctx, n.w_up, h, f16));
    return residual ? ggml_add(ctx, y, residual) : y;
}

}


static void mp_quiet_log(ggml_log_level level, const char * text, void *) {
    static const bool verbose = getenv("MP_GGML_VERBOSE") != nullptr;
    if (verbose || level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

ModernProstModel::ModernProstModel(const std::string & gguf_path, const std::string & device) {
    ggml_log_set(mp_quiet_log, nullptr);
    ggml_context * meta = nullptr;
    gguf_init_params gp = {  true,  &meta };
    gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (gguf == nullptr) {
        throw std::runtime_error("ModernProst: cannot read gguf: " + gguf_path);
    }

    auto u32 = [&](const char * k) {
        const int i = gguf_find_key(gguf, k);
        if (i < 0) throw std::runtime_error(std::string("ModernProst: missing key ") + k);
        return gguf_get_val_u32(gguf, i);
    };
    auto f32 = [&](const char * k) {
        const int i = gguf_find_key(gguf, k);
        if (i < 0) throw std::runtime_error(std::string("ModernProst: missing key ") + k);
        return gguf_get_val_f32(gguf, i);
    };

    hp.n_layer        = u32("modernprost.block_count");
    hp.n_embd         = u32("modernprost.embedding_length");
    hp.n_head         = u32("modernprost.attention.head_count");
    hp.head_dim       = u32("modernprost.attention.key_length");
    hp.n_ff           = u32("modernprost.feed_forward_length");
    hp.sliding_window = u32("modernprost.sliding_window");
    hp.noble_rank     = u32("modernprost.noble_rank");
    hp.rope_global    = f32("modernprost.global_rope_theta");
    hp.rope_local     = f32("modernprost.local_rope_theta");
    hp.eps            = f32("modernprost.attention.layer_norm_epsilon");

    {
        const int i = gguf_find_key(gguf, "modernprost.attn_layer_types");
        if (i < 0) throw std::runtime_error("ModernProst: missing attn_layer_types");
        const int32_t * d = (const int32_t *) gguf_get_arr_data(gguf, i);
        hp.layer_is_sliding.assign(d, d + gguf_get_arr_n(gguf, i));
    }
    {
        const int i = gguf_find_key(gguf, "modernprost.vocab");
        if (i < 0) throw std::runtime_error("ModernProst: missing vocab");
        for (size_t j = 0; j < gguf_get_arr_n(gguf, i); ++j) {
            vocab[gguf_get_arr_str(gguf, i, j)] = (int32_t) j;
        }
    }

    backend = nullptr;
#ifdef GGML_USE_CUDA
    if (device != "cpu") {
        backend = ggml_backend_cuda_init(0);
        if (backend == nullptr) {
            fprintf(stderr, "ModernProst: CUDA init failed, falling back to CPU\n");
        }
    }
#else
    (void) device;
#endif
    if (backend == nullptr) {
        backend = ggml_backend_cpu_init();
        gpu = false;
    } else {
        gpu = true;
    }

    const bool upcast = !gpu && getenv("MP_CPU_KEEP_F16") == nullptr;
    const int n_tensors = gguf_get_n_tensors(gguf);
    ggml_init_params ip = { ggml_tensor_overhead() * (n_tensors + 1), nullptr, true };
    ctx_w = ggml_init(ip);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * src  = ggml_get_tensor(meta, name);
        const ggml_type t  = (upcast && src->type == GGML_TYPE_F16) ? GGML_TYPE_F32 : src->type;
        ggml_tensor * dst  = ggml_new_tensor(ctx_w, t, GGML_MAX_DIMS, src->ne);
        ggml_set_name(dst, name);
    }
    buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    if (buf_w == nullptr) throw std::runtime_error("ModernProst: cannot allocate weights");

    {
        FILE * f = fopen(gguf_path.c_str(), "rb");
        if (f == nullptr) throw std::runtime_error("ModernProst: cannot open " + gguf_path);
        const size_t data_off = gguf_get_data_offset(gguf);
        std::vector<uint8_t> tmp;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf, i);
            ggml_tensor * dst = ggml_get_tensor(ctx_w, name);
            const size_t off  = data_off + gguf_get_tensor_offset(gguf, i);
            const size_t sz   = gguf_get_tensor_size(gguf, i);
            tmp.resize(sz);
            if (fseek(f, (long) off, SEEK_SET) != 0 || fread(tmp.data(), 1, sz, f) != sz) {
                fclose(f);
                throw std::runtime_error(std::string("ModernProst: short read on ") + name);
            }
            if (gguf_get_tensor_type(gguf, i) == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
                std::vector<float> up(ggml_nelements(dst));
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) tmp.data(), up.data(), (int64_t) up.size());
                ggml_backend_tensor_set(dst, up.data(), 0, ggml_nbytes(dst));
            } else {
                ggml_backend_tensor_set(dst, tmp.data(), 0, sz);
            }
        }
        fclose(f);
    }
    ggml_free(meta);

    tok_embd   = need(ctx_w, "tok_embd.weight");
    embd_norm  = need(ctx_w, "embd_norm.weight");
    final_norm = need(ctx_w, "final_norm.weight");
    head_3di   = need(ctx_w, "head_3di.weight");
    head_12st  = need(ctx_w, "head_12st.weight");
    layers.resize(hp.n_layer);
    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        ModernProstLayer & L = layers[il];
        L.attn_norm = il == 0 ? nullptr : need(ctx_w, b + "attn_norm.weight");
        L.mlp_norm  = need(ctx_w, b + "mlp_norm.weight");
        L.ffn_up    = need(ctx_w, b + "ffn_up.weight");
        L.ffn_down  = need(ctx_w, b + "ffn_down.weight");
        L.qkv.linear = need(ctx_w, b + "attn_qkv.weight");
        L.out.linear = need(ctx_w, b + "attn_out.weight");
        for (int k = 0; k < 2; ++k) {
            ModernProstNoble & n = k == 0 ? L.qkv : L.out;
            const std::string t = k == 0 ? "qkv" : "out";
            n.w_down = need(ctx_w, b + "noble_" + t + "_down.weight");
            n.w_up   = need(ctx_w, b + "noble_" + t + "_up.weight");
            n.m      = need(ctx_w, b + "noble_" + t + "_M");
            n.omega1 = need(ctx_w, b + "noble_" + t + "_omega1");
            n.phi1   = need(ctx_w, b + "noble_" + t + "_phi1");
            n.omega2 = need(ctx_w, b + "noble_" + t + "_omega2");
            n.phi2   = need(ctx_w, b + "noble_" + t + "_phi2");
        }
    }

    folded = getenv("MP_NO_FOLD") == nullptr;
    if (folded) {
        auto get = [](ggml_tensor * t) {
            std::vector<float> v(ggml_nelements(t));
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<ggml_fp16_t> h(v.size());
                ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
                ggml_fp16_to_fp32_row(h.data(), v.data(), (int64_t) v.size());
            }
            return v;
        };
        auto put = [](ggml_tensor * t, const std::vector<float> & v) {
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
            } else {
                std::vector<ggml_fp16_t> h(v.size());
                ggml_fp32_to_fp16_row(v.data(), h.data(), (int64_t) v.size());
                ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
            }
        };
        auto scale = [&](ggml_tensor * W, const std::vector<float> & s, bool by_input) {
            std::vector<float> w = get(W);
            const int64_t n_in = W->ne[0], n_out = ggml_nelements(W) / n_in;
            if ((int64_t) s.size() != (by_input ? n_in : n_out)) {
                throw std::runtime_error(std::string("ModernProst: cannot fold scale into ") + ggml_get_name(W));
            }
            for (int64_t o = 0; o < n_out; ++o) {
                for (int64_t i = 0; i < n_in; ++i) {
                    w[o * n_in + i] *= by_input ? s[i] : s[o];
                }
            }
            put(W, w);
        };
        for (ModernProstLayer & L : layers) {
            if (L.attn_norm) {
                const std::vector<float> g = get(L.attn_norm);
                scale(L.qkv.linear, g, true);
                if (L.qkv.w_down) scale(L.qkv.w_down, g, true);
            }
            scale(L.ffn_up, get(L.mlp_norm), true);
            for (ModernProstNoble * n : { &L.qkv, &L.out }) {
                if (n->w_down == nullptr) continue;
                scale(n->w_down, get(n->omega1), false);
                scale(n->m,      get(n->omega2), false);
            }
        }
        const std::vector<float> gf = get(final_norm);
        scale(head_3di, gf, true);
        scale(head_12st, gf, true);
    }
}

ModernProstModel::~ModernProstModel() {
    if (buf_w)   ggml_backend_buffer_free(buf_w);
    if (ctx_w)   ggml_free(ctx_w);
    if (backend) ggml_backend_free(backend);
    if (gguf)    gguf_free(gguf);
}


void mp_phase_report();
ModernProst::ModernProst(ModernProstModel & m, int threads) : model(m), n_threads(threads) {
    static const bool registered = getenv("MP_PHASEPROF") != nullptr && atexit(mp_phase_report) == 0;
    (void) registered;
}

ModernProst::~ModernProst() {
    for (auto & kv : cache) {
        if (kv.second.galloc) ggml_gallocr_free(kv.second.galloc);
        if (kv.second.ctx)    ggml_free(kv.second.ctx);
    }
}

int ModernProst::bucket_for(int n) {
    static const int fixed = getenv("MP_PAD") ? atoi(getenv("MP_PAD")) : 0;
    int step;
    if (fixed == 64) {
        step = n <= 512 ? 64 : (n <= 2048 ? 256 : 1024);
    } else {
        step = fixed > 0 ? fixed : 8;
        while (fixed <= 0 && step * 32 < n) step *= 2;
    }
    return ((n + step - 1) / step) * step;
}

std::vector<int32_t> ModernProst::tokenize(const std::string & aa) const {
    std::vector<int32_t> ids;
    ids.reserve(aa.size());
    auto unk = model.vocab.find("<unk>");
    const int32_t unk_id = unk == model.vocab.end() ? 2 : unk->second;
    for (char c : aa) {
        char up = (char) toupper((unsigned char) c);
        auto it = model.vocab.find(std::string(1, up));
        ids.push_back(it == model.vocab.end() ? unk_id : it->second);
    }
    return ids;
}

ggml_cgraph * ModernProst::build_graph(ModernProst::Cached & c, int n_tokens, int n_seq) {
    ggml_context * ctx = c.ctx;
    const ModernProstHparams & hp = model.hp;
    const int n_embd    = (int) hp.n_embd;
    const int n_head    = (int) hp.n_head;
    const int head_dim  = (int) hp.head_dim;
    const float kq_scale = 1.0f / sqrtf((float) head_dim);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    const int n_tot = n_tokens * n_seq;
    c.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tot);
    ggml_set_input(c.inp_tokens);
    c.inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tot);
    ggml_set_input(c.inp_pos);
    const int hd = (int) hp.head_dim;
    const bool hd_ok = (hd == 32 || hd == 64 || hd == 80 || hd == 96 || hd == 112
                        || hd == 128 || hd == 192 || hd == 256);
    const bool pad_flash = !hd_ok && hd < 64 && getenv("MP_PAD_FLASH") != nullptr;
    const bool use_flash = (hd_ok || pad_flash) && getenv("MP_NO_FLASH") == nullptr;
    const bool act16 = model.gpu && use_flash && !pad_flash && getenv("MP_NO_F16_ACT") == nullptr;
    const ggml_type mtype = use_flash ? GGML_TYPE_F16 : GGML_TYPE_F32;
    c.mask_f16 = use_flash;
    c.mask_full = ggml_new_tensor_4d(ctx, mtype, n_tokens, n_tokens, 1, n_seq);
    ggml_set_input(c.mask_full);
    c.mask_slide = ggml_new_tensor_4d(ctx, mtype, n_tokens, n_tokens, 1, n_seq);
    ggml_set_input(c.mask_slide);

    ggml_tensor * cur = ggml_get_rows(ctx, model.tok_embd, c.inp_tokens);
    cur = layer_norm(ctx, cur, model.embd_norm, hp.eps);
    if (act16) cur = ggml_cast(ctx, cur, GGML_TYPE_F16);

    const bool folded = model.folded;
    static const bool old_layout = getenv("MP_OLD_LAYOUT") != nullptr;
    const bool new_path = !old_layout && use_flash && !pad_flash;
    for (uint32_t il = 0; il < hp.n_layer; ++il) {
        const ModernProstLayer & L = model.layers[il];
        const bool sliding = model.hp.layer_is_sliding[il] != 0;

        ggml_tensor * residual = cur;
        ggml_tensor * x = L.attn_norm ? layer_norm(ctx, cur, folded ? nullptr : L.attn_norm, hp.eps) : cur;

        ggml_tensor * qkv = noble(ctx, x, L.qkv, act16, folded);
        const float theta = sliding ? hp.rope_local : hp.rope_global;
        ggml_tensor * q, * k, * v;
        if (new_path) {
            const size_t es = ggml_element_size(qkv);
            ggml_tensor * qk = ggml_view_3d(ctx, qkv, head_dim, 2 * n_head, n_tot,
                                            head_dim * es, qkv->nb[1], 0);
            qk = ggml_rope_ext(ctx, qk, c.inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                               0, theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            const size_t qes = ggml_element_size(qk);
            q = ggml_view_4d(ctx, qk, head_dim, n_head, n_tokens, n_seq,
                             head_dim * qes, qk->nb[1] * 2 * n_head, qk->nb[1] * 2 * n_head * n_tokens, 0);
            k = ggml_view_4d(ctx, qk, head_dim, n_head, n_tokens, n_seq,
                             head_dim * qes, qk->nb[1] * 2 * n_head, qk->nb[1] * 2 * n_head * n_tokens,
                             (size_t) n_head * head_dim * qes);
            v = ggml_view_4d(ctx, qkv, head_dim, n_head, n_tokens, n_seq,
                             head_dim * es, qkv->nb[1], qkv->nb[1] * n_tokens, 2ull * n_embd * es);
        } else {
            q = ggml_view_2d(ctx, qkv, n_embd, n_tot, qkv->nb[1], 0);
            k = ggml_view_2d(ctx, qkv, n_embd, n_tot, qkv->nb[1], 1ull * n_embd * ggml_element_size(qkv));
            v = ggml_view_2d(ctx, qkv, n_embd, n_tot, qkv->nb[1], 2ull * n_embd * ggml_element_size(qkv));
            q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_head, n_tot);
            k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_head, n_tot);
            q = ggml_rope_ext(ctx, q, c.inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                              0, theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            k = ggml_rope_ext(ctx, k, c.inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                              0, theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        }

        ggml_tensor * kqv;
        ggml_tensor * mask = sliding ? c.mask_slide : c.mask_full;
        if (new_path) {
            ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
            qp = act16 ? ggml_cast(ctx, qp, GGML_TYPE_F32) : ggml_cont(ctx, qp);
            kqv = ggml_flash_attn_ext(ctx, qp, ggml_permute(ctx, k, 0, 2, 1, 3),
                                      ggml_permute(ctx, v, 0, 2, 1, 3), mask, kq_scale, 0.0f, 0.0f);
            kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tot);
        } else {
        if (act16) q = ggml_cast(ctx, q, GGML_TYPE_F32);
        q = ggml_reshape_4d(ctx, q, head_dim, n_head, n_tokens, n_seq);
        k = ggml_reshape_4d(ctx, k, head_dim, n_head, n_tokens, n_seq);
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        if (use_flash) {
            ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx,
                                   ggml_reshape_4d(ctx, ggml_cont(ctx, v), head_dim, n_head, n_tokens, n_seq),
                                   0, 2, 1, 3));
            ggml_tensor * qf = qp, * kf = kp, * vf = vp;
            if (pad_flash) {
                const int padn = 64 - head_dim;
                qf = ggml_pad(ctx, ggml_cont(ctx, qp), padn, 0, 0, 0);
                kf = ggml_pad(ctx, kf, padn, 0, 0, 0);
                vf = ggml_pad(ctx, vf, padn, 0, 0, 0);
            }
            kqv = ggml_flash_attn_ext(ctx, qf, kf, vf, mask, kq_scale, 0.0f, 0.0f);
            if (pad_flash) {
                const size_t es = sizeof(float);
                kqv = ggml_cont(ctx, ggml_view_4d(ctx, kqv, head_dim, n_head, n_tokens, n_seq,
                                                  64 * es, 64 * es * n_head,
                                                  64 * es * n_head * n_tokens, 0));
            }
            kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tot);
        } else {
            ggml_tensor * kq = mp_mul_mat(ctx, kp, qp);
            kq = ggml_soft_max_ext(ctx, kq, mask, kq_scale, 0.0f);
            ggml_tensor * v4 = ggml_reshape_4d(ctx, ggml_cont(ctx, v), head_dim, n_head, n_tokens, n_seq);
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 2, 0, 3));
            kqv = mp_mul_mat(ctx, vt, kq);
            kqv = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), n_embd, n_tot);
        }
        }

        if (act16) kqv = ggml_cast(ctx, kqv, GGML_TYPE_F16);
        cur = old_layout ? ggml_add(ctx, residual, noble(ctx, kqv, L.out, act16, folded))
                         : noble(ctx, kqv, L.out, act16, folded, residual);

        ggml_tensor * res2 = cur;
        ggml_tensor * h = layer_norm(ctx, cur, folded ? nullptr : L.mlp_norm, hp.eps);
        h = mp_mul_mat(ctx, L.ffn_up, h, act16);
        const int n_ff = (int) hp.n_ff;
        if (getenv("MP_NO_GLU") == nullptr) {
            h = ggml_swiglu_swapped(ctx, h);
        } else {
            ggml_tensor * up   = ggml_view_2d(ctx, h, n_ff, n_tot, h->nb[1], 0);
            ggml_tensor * gate = ggml_view_2d(ctx, h, n_ff, n_tot, h->nb[1], 1ull * n_ff * ggml_element_size(h));
            h = ggml_mul(ctx, ggml_silu(ctx, ggml_cont(ctx, gate)), ggml_cont(ctx, up));
        }
        cur = ggml_add(ctx, res2, mp_mul_mat(ctx, L.ffn_down, h, act16));
    }

    cur = layer_norm(ctx, cur, folded ? nullptr : model.final_norm, hp.eps);
    c.out_3di  = mp_mul_mat(ctx, model.head_3di,  cur);
    c.out_12st = mp_mul_mat(ctx, model.head_12st, cur);
    ggml_set_output(c.out_3di);
    ggml_set_output(c.out_12st);
    ggml_build_forward_expand(gf, c.out_3di);
    ggml_build_forward_expand(gf, c.out_12st);
    return gf;
}

ModernProst::Cached * ModernProst::get_graph(int n_tokens, int n_seq, int slot) {
    auto key = std::make_tuple(n_tokens, n_seq, slot);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return &it->second;
    }
    Cached c;
    c.n_tokens = n_tokens;
    c.n_seq = n_seq;
    const size_t mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    ggml_init_params ip = { mem, nullptr, true };
    c.ctx = ggml_init(ip);
    c.gf = build_graph(c, n_tokens, n_seq);
    c.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!ggml_gallocr_reserve(c.galloc, c.gf)) {
        ggml_gallocr_free(c.galloc);
        ggml_free(c.ctx);
        return nullptr;
    }
    auto res = cache.emplace(key, c);
    return &res.first->second;
}

void ModernProst::clear_cache() {
    for (auto & kv : cache) {
        if (kv.second.galloc) ggml_gallocr_free(kv.second.galloc);
        if (kv.second.ctx)    ggml_free(kv.second.ctx);
    }
    cache.clear();
}

bool ModernProst::run(const std::string & aa,
                      std::vector<float> & logits_3di,
                      std::vector<float> & logits_12st) {
    std::vector<std::vector<float>> b3, b12;
    if (!run_batch({aa}, b3, b12)) return false;
    logits_3di  = std::move(b3[0]);
    logits_12st = std::move(b12[0]);
    return true;
}

static double g_ph[8] = {0}; static const char * g_phn[8] =
    {"tokenize+alloc", "mask build", "upload", "wait for GPU", "get+split", "records", "launch (host)", ""};
static bool mp_phaseprof() { static const bool v = getenv("MP_PHASEPROF") != nullptr; return v; }
static inline double mp_now() { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void mp_phase_reset() { for (double & x : g_ph) x = 0; }
void mp_phase_report() {
    if (!mp_phaseprof()) return;
    double t = 0; for (int i = 0; i < 7; ++i) t += g_ph[i];
    fprintf(stderr, "[phase] total %.0f ms\n", t);
    for (int i = 0; i < 7; ++i) fprintf(stderr, "[phase]   %-16s %8.0f ms  %5.1f%%\n", g_phn[i], g_ph[i], 100*g_ph[i]/t);
}

bool ModernProst::stage(const std::vector<std::string> & seqs, Staged & st, int slot) {
    double _t = mp_now();
    st.B = (int) seqs.size();
    if (st.B == 0) return false;
    st.ids.assign(st.B, {});
    int maxlen = 0;
    for (int b = 0; b < st.B; ++b) {
        st.ids[b] = tokenize(seqs[b]);
        maxlen = std::max(maxlen, (int) st.ids[b].size());
    }
    if (maxlen == 0) return false;
    st.n = bucket_for(maxlen);
    const int n = st.n, B = st.B;

    st.c = get_graph(n, B, slot);
    if (st.c == nullptr) return false;

    const size_t tot = (size_t) n * B;
    st.tok.assign(tot, 0);
    st.pos.resize(tot);
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < n; ++i) {
            const size_t o = (size_t) b * n + i;
            st.tok[o] = i < (int) st.ids[b].size() ? st.ids[b][i] : 0;
            st.pos[o] = i;
        }
    }
    { double t2 = mp_now(); g_ph[0] += t2 - _t; _t = t2; }

    const int w = (int) model.hp.sliding_window;
    auto build = [&](auto * pf, auto * psl, auto zero, auto ninf) {
        for (int b = 0; b < B; ++b) {
            const int L = (int) st.ids[b].size();
            for (int i = 0; i < n; ++i) {
                auto * rf = pf  + ((size_t) b * n + i) * n;
                auto * rs = psl + ((size_t) b * n + i) * n;
                std::fill(rf, rf + L, zero);
                std::fill(rf + L, rf + n, ninf);
                rf[i] = zero;
                std::fill(rs, rs + n, ninf);
                const int lo = std::max(0, i - w), hi = std::min(L - 1, i + w);
                if (lo <= hi) std::fill(rs + lo, rs + hi + 1, zero);
                rs[i] = zero;
            }
        }
    };
    if (st.c->mask_f16) {
        st.m16f.resize(tot * n); st.m16s.resize(tot * n);
        build(st.m16f.data(), st.m16s.data(), (uint16_t) ggml_fp32_to_fp16(0.0f), (uint16_t) ggml_fp32_to_fp16(-INFINITY));
    } else {
        st.m32f.resize(tot * n); st.m32s.resize(tot * n);
        build(st.m32f.data(), st.m32s.data(), 0.0f, -INFINITY);
    }
    g_ph[1] += mp_now() - _t;
    return true;
}

bool ModernProst::upload(Staged & st) {
    const double _t = mp_now();
    Cached * c = st.c;
    if (!ggml_gallocr_alloc_graph(c->galloc, c->gf)) return false;
    ggml_backend_tensor_set(c->inp_tokens, st.tok.data(), 0, st.tok.size() * sizeof(int32_t));
    ggml_backend_tensor_set(c->inp_pos,    st.pos.data(), 0, st.pos.size() * sizeof(int32_t));
    if (c->mask_f16) {
        ggml_backend_tensor_set(c->mask_full,  st.m16f.data(), 0, st.m16f.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(c->mask_slide, st.m16s.data(), 0, st.m16s.size() * sizeof(uint16_t));
    } else {
        ggml_backend_tensor_set(c->mask_full,  st.m32f.data(), 0, st.m32f.size() * sizeof(float));
        ggml_backend_tensor_set(c->mask_slide, st.m32s.data(), 0, st.m32s.size() * sizeof(float));
    }
    g_ph[2] += mp_now() - _t;
    return true;
}

bool ModernProst::compute_async(Staged & st) {
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
    const double _t = mp_now();
    const bool ok = ggml_backend_graph_compute_async(model.backend, st.c->gf) == GGML_STATUS_SUCCESS;
    g_ph[6] += mp_now() - _t;
    return ok;
}

bool ModernProst::collect(Staged & st, std::vector<std::vector<float>> & logits_3di,
                          std::vector<std::vector<float>> & logits_12st) {
    double _t = mp_now();
    ggml_backend_synchronize(model.backend);
    { double t2 = mp_now(); g_ph[3] += t2 - _t; _t = t2; }
    const size_t tot = (size_t) st.n * st.B;
    std::vector<float> a3(tot * 20), a12(tot * 12);
    ggml_backend_tensor_get(st.c->out_3di,  a3.data(),  0, a3.size()  * sizeof(float));
    ggml_backend_tensor_get(st.c->out_12st, a12.data(), 0, a12.size() * sizeof(float));
    logits_3di.assign(st.B, {});
    logits_12st.assign(st.B, {});
    for (int b = 0; b < st.B; ++b) {
        const size_t L = st.ids[b].size();
        const size_t o = (size_t) b * st.n;
        logits_3di[b].assign (a3.begin()  + o * 20, a3.begin()  + (o + L) * 20);
        logits_12st[b].assign(a12.begin() + o * 12, a12.begin() + (o + L) * 12);
    }
    g_ph[4] += mp_now() - _t;
    return true;
}

bool ModernProst::run_batch(const std::vector<std::string> & seqs,
                            std::vector<std::vector<float>> & logits_3di,
                            std::vector<std::vector<float>> & logits_12st) {
    Staged st;
    if (!stage(seqs, st, 0) || !upload(st) || !compute_async(st)) return false;
    return collect(st, logits_3di, logits_12st);
}

bool ModernProst::predict(const std::string & aa, std::string & three_di,
                          std::string & twelve_st, std::string & packed) {
    std::vector<float> l3, l12;
    if (!run(aa, l3, l12)) return false;
    const size_t n = aa.size();
    three_di.resize(n); twelve_st.resize(n); packed.resize(n);
    for (size_t t = 0; t < n; ++t) {
        int a3 = 0;  for (int i = 1; i < 20; ++i) if (l3 [t*20+i] > l3 [t*20+a3]) a3 = i;
        int a12 = 0; for (int i = 1; i < 12; ++i) if (l12[t*12+i] > l12[t*12+a12]) a12 = i;
        three_di[t]  = MODERNPROST_3DI_ALPHABET[a3];
        twelve_st[t] = MODERNPROST_12ST_ALPHABET[a12];
        packed[t]    = (char) (a3 * 12 + a12);
    }
    return true;
}


namespace {

const float BACKPROB_3DI[20] = {
    0.0489372f, 0.0306991f, 0.101049f,  0.0329671f, 0.0276149f,
    0.0416262f, 0.0452521f, 0.030876f,  0.0297251f, 0.0607036f,
    0.0150238f, 0.0215826f, 0.0783843f, 0.0512926f, 0.0264886f,
    0.0610702f, 0.0201311f, 0.215998f,  0.0310265f, 0.0295417f
};
const float BACKPROB_12ST[20] = {
    0.0412689151029924f, 0.0231671389192358f, 0.0599445079508294f,
    0.0796249244089918f, 0.113519805597932f,  0.0845763487053319f,
    0.124117111554957f,  0.105288913376871f,  0.150162613848461f,
    0.0882703257027577f, 0.0471395802906669f, 0.082901814540972f,
    1e-06f, 1e-06f, 1e-06f, 1e-06f, 1e-06f, 1e-06f, 1e-06f, 1e-06f
};
const float BIT_FACTOR = 8.0f;

void logits_to_record(const float * logits, size_t len, int k,
                      const float * bg, std::vector<uint8_t> & out) {
    out.assign(len * 25, 0);
    std::vector<float> p(20, 0.0f);
    for (size_t t = 0; t < len; ++t) {
        const float * l = logits + t * k;
        float mx = l[0];
        for (int i = 1; i < k; ++i) mx = std::max(mx, l[i]);
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) { p[i] = expf(l[i] - mx); sum += p[i]; }
        int best = 0;
        for (int i = 0; i < k; ++i) { p[i] /= sum; if (p[i] > p[best]) best = i; }
        for (int i = k; i < 20; ++i) p[i] = 0.0f;

        uint8_t * rec = out.data() + t * 25;
        for (int i = 0; i < 20; ++i) {
            float v = -128.0f;
            const float odds = p[i] / bg[i];
            if (odds > 0.0f) {
                v = log2f(odds) * BIT_FACTOR;
                v = v < 0.0f ? v - 0.5f : v + 0.5f;
            }
            if (v < -128.0f) v = -128.0f;
            if (v >  127.0f) v =  127.0f;
            rec[i] = (uint8_t) (int8_t) (int) v;
        }
        rec[20] = (uint8_t) best;
        rec[21] = (uint8_t) best;
        rec[22] = 1;
        rec[23] = 0;
        rec[24] = 0;
    }
}

}

bool ModernProst::predict_profiles(const std::string & aa,
                                   std::vector<uint8_t> & rec_3di,
                                   std::vector<uint8_t> & rec_12st) {
    std::vector<float> l3, l12;
    if (!run(aa, l3, l12)) return false;
    const size_t len = aa.size();
    logits_to_record(l3.data(),  len, 20, BACKPROB_3DI,  rec_3di);
    logits_to_record(l12.data(), len, 12, BACKPROB_12ST, rec_12st);
    return true;
}

bool ModernProst::predict_profiles_batch(const std::vector<std::string> & seqs,
                                         std::vector<std::vector<uint8_t>> & rec_3di,
                                         std::vector<std::vector<uint8_t>> & rec_12st) {
    std::vector<std::vector<float>> l3, l12;
    if (!run_batch(seqs, l3, l12)) return false;
    const double _t = mp_now();
    rec_3di.assign(seqs.size(), {});
    rec_12st.assign(seqs.size(), {});
    for (size_t b = 0; b < seqs.size(); ++b) {
        logits_to_record(l3[b].data(),  seqs[b].size(), 20, BACKPROB_3DI,  rec_3di[b]);
        logits_to_record(l12[b].data(), seqs[b].size(), 12, BACKPROB_12ST, rec_12st[b]);
    }
    g_ph[5] += mp_now() - _t;
    return true;
}

bool ModernProst::predict_profiles_many(const std::vector<std::string> & seqs,
                                        std::vector<std::vector<uint8_t>> & rec_3di,
                                        std::vector<std::vector<uint8_t>> & rec_12st,
                                        const ModernProstBatchPolicy & pol,
                                        const std::function<void(size_t)> & on_done) {
    const size_t N = seqs.size();
    rec_3di.assign(N, {});
    rec_12st.assign(N, {});
    if (N == 0) return true;

    struct Item { size_t owner; int chunk; std::string seq; };
    std::vector<Item> items;
    std::vector<int> nchunks(N, 1);
    for (size_t i = 0; i < N; ++i) {
        const int L = (int) seqs[i].size();
        if (L == 0) continue;
        if (pol.chunk_len > 0 && L > pol.chunk_len) {
            const int n = (L + pol.chunk_len - 1) / pol.chunk_len;
            const int base = L / n, rem = L % n;
            nchunks[i] = n;
            int start = 0;
            for (int c = 0; c < n; ++c) {
                const int size = base + (c < rem ? 1 : 0);
                items.push_back({ i, c, seqs[i].substr(start, size) });
                start += size;
            }
        } else {
            items.push_back({ i, 0, seqs[i] });
        }
    }
    std::sort(items.begin(), items.end(),
              [](const Item & a, const Item & b) { return a.seq.size() > b.seq.size(); });

    std::vector<std::vector<uint8_t>> part3(items.size()), part12(items.size());

    std::vector<std::vector<size_t>> batches;
    {
        std::vector<size_t> cur; long residues = 0;
        for (size_t k = 0; k < items.size(); ++k) {
            cur.push_back(k);
            residues += (long) items[k].seq.size();
            if ((int) cur.size() >= pol.batch_size || residues >= pol.max_residues) {
                batches.push_back(std::move(cur)); cur.clear(); residues = 0;
            }
        }
        if (!cur.empty()) batches.push_back(std::move(cur));
    }

    const bool pipelined = getenv("MP_NO_PIPELINE") == nullptr;
    std::vector<int> left = nchunks;
    if (on_done) {
        size_t empty = 0;
        for (size_t i = 0; i < N; ++i) empty += seqs[i].empty();
        if (empty) on_done(empty);
    }
    auto to_records = [&](const std::vector<size_t> & idx,
                          std::vector<std::vector<float>> & l3, std::vector<std::vector<float>> & l12) {
        const double _t = mp_now();
        size_t finished = 0;
        for (size_t q = 0; q < idx.size(); ++q) {
            const size_t k = idx[q];
            logits_to_record(l3[q].data(),  items[k].seq.size(), 20, BACKPROB_3DI,  part3[k]);
            logits_to_record(l12[q].data(), items[k].seq.size(), 12, BACKPROB_12ST, part12[k]);
            finished += --left[items[k].owner] == 0;
        }
        g_ph[5] += mp_now() - _t;
        if (on_done && finished) on_done(finished);
    };
    Staged prev; const std::vector<size_t> * prev_idx = nullptr;
    for (size_t bi = 0; bi < batches.size(); ++bi) {
        std::vector<std::string> seqs_b;
        for (size_t k : batches[bi]) seqs_b.push_back(items[k].seq);
        Staged cur;
        if (!stage(seqs_b, cur, pipelined ? (int) (bi & 1) : 0)) return false;
        if (!pipelined) {
            std::vector<std::vector<float>> l3, l12;
            if (!upload(cur) || !compute_async(cur) || !collect(cur, l3, l12)) return false;
            to_records(batches[bi], l3, l12);
            continue;
        }
        if (!upload(cur)) return false;
        std::vector<std::vector<float>> l3, l12;
        if (prev_idx && !collect(prev, l3, l12)) return false;
        if (!compute_async(cur)) return false;
        if (prev_idx) to_records(*prev_idx, l3, l12);
        prev = std::move(cur); prev_idx = &batches[bi];
    }
    if (prev_idx) {
        std::vector<std::vector<float>> l3, l12;
        if (!collect(prev, l3, l12)) return false;
        to_records(*prev_idx, l3, l12);
    }

    std::vector<std::vector<std::pair<int, size_t>>> byOwner(N);
    for (size_t k = 0; k < items.size(); ++k) byOwner[items[k].owner].push_back({items[k].chunk, k});
    for (size_t i = 0; i < N; ++i) {
        auto & v = byOwner[i];
        std::sort(v.begin(), v.end());
        for (auto & pr : v) {
            rec_3di[i].insert (rec_3di[i].end(),  part3 [pr.second].begin(), part3 [pr.second].end());
            rec_12st[i].insert(rec_12st[i].end(), part12[pr.second].begin(), part12[pr.second].end());
        }
    }
    return true;
}
