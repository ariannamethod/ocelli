/*
 * ocelli.c — the eyes of Yent: SmolVLM-class inference on notorch, pure C.
 *
 * Image + prompt -> text, no cloud/API. Built on vendored notorch. The vision
 * tower (SigLIP), the pixel-shuffle connector and the preprocessing live in
 * vision.c; this file is the orchestrator, the llama-family text decoder and
 * the splice point where visual embeddings enter the token stream.
 *
 * Raw-token-id mode (--ids) is kept for isolated verification against a
 * reference runtime, independent of our tokenizer.
 *
 * Build: make      (see Makefile for the portable / ASan variants)
 * Run:   ./ocelli <model.gguf> [--ids "1 2 3 ..."] [--prompt "text"] [-n N]
 *        ./eye <image>          (the CLI wrapper)
 *
 * Decoding is greedy (argmax) — deterministic, to match llama.cpp --temp 0.
 */

#include "gguf.h"
#include "notorch.h"
#include "bpe.h"
#include "vision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ── math (mirrors examples/infer_llama.c, the verified llama decode path) ──── */

// C[m,n] = A[m,k] @ B^T[n,k]   (B is GGUF weight layout: [out, in] row-major)
static void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

/* ── f16 weights (half RAM): each matmul weight is f16 (exact GGUF value) or f32.
 *    Lazy-dequant f16 -> reused f32 scratch right before cblas (portable BLAS). ── */
static inline float f16_to_f32_smol(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, r;
    if (exp == 0) {
        if (mant == 0) { r = sign; }
        else { exp = 127 - 15 + 1; while (!(mant & 0x400)) { mant <<= 1; exp--; }
               mant &= 0x3FF; r = sign | (exp << 23) | (mant << 13); }
    } else if (exp == 0x1F) { r = sign | 0x7F800000 | (mant << 13); }
    else { r = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    float f; memcpy(&f, &r, 4); return f;
}
typedef struct { const uint16_t *f16; const float *f32; } wt;   /* exactly one set */

static wt load_wt(gguf_file *gf, const char *name) {
    wt w = {NULL, NULL};
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return w;
    w.f16 = gguf_load_f16(gf, ti);          /* non-NULL iff tensor is F16 */
    if (!w.f16) w.f32 = gguf_dequant(gf, ti);
    return w;
}
static int wt_ok(wt w) { return w.f16 || w.f32; }

static float *g_wscratch = NULL; static long g_wscap = 0;
static const float *materialize(wt w, long n) {        /* -> f32 view (scratch if f16) */
    if (w.f32) return w.f32;
    if (!w.f16) return NULL;
    if (n > g_wscap) { free(g_wscratch); g_wscratch = (float*)malloc(n * sizeof(float));
                       g_wscap = g_wscratch ? n : 0; }   /* cap only on success -> retry works */
    if (!g_wscratch) return NULL;                        /* OOM: no NULL write */
    gguf_f16_to_f32_n(w.f16, g_wscratch, n);
    return g_wscratch;
}

static void mm_t(float *C, const float *A, const float *B, int m, int k, int n);  /* fwd */
/* C[m,n] = A[m,k] @ W[n,k]^T with a wt weight (lazy-dequant) */
static void mm_t_w(float *C, const float *A, wt W, int m, int k, int n) {
    mm_t(C, A, materialize(W, (long)n * k), m, k, n);
}
/* lm_head matvec: logits[v] = sum_k x[k]*W[v,k], W f16-or-f32, no big scratch (per-element) */
static void lmhead(wt W, const float *x, float *logits, int vocab, int E) {
    if (W.f32) { mm_t(logits, x, W.f32, 1, E, vocab); return; }
    const uint16_t *w = W.f16;
    for (int v = 0; v < vocab; v++) {
        const uint16_t *row = w + (long)v * E;
        float s = 0;
        for (int k = 0; k < E; k++) s += x[k] * f16_to_f32_smol(row[k]);
        logits[v] = s;
    }
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

// RoPE interleaved (2i,2i+1) — correct for GGUF-converted llama (weights permuted at convert)
static void rope(float *x, int pos, int head_dim, float freq_base) {
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
        float angle = pos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float x0 = x[2*i], x1 = x[2*i+1];
        x[2*i]   = x0 * cs - x1 * sn;
        x[2*i+1] = x0 * sn + x1 * cs;
    }
}

static void add_bias(float *x, const float *bias, int n) {
    if (bias) for (int i = 0; i < n; i++) x[i] += bias[i];
}

/* ── model ──────────────────────────────────────────────────────────────────── */

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    float rope_base, rms_eps;
    int has_output_weight;

    wt tok_emb, out_weight;        /* f16-or-f32 */
    float *out_norm;
    struct {
        float *attn_norm;
        wt wq, wk, wv, wo;         /* f16-or-f32 */
        float *q_bias, *k_bias, *v_bias;
        float *ffn_norm;
        wt wgate, wup, wdown;      /* f16-or-f32 */
    } layers[];
} llama_model;

static llama_model* llama_load(gguf_file* gf) {
    int nl = gf->n_layers;
    llama_model* m = (llama_model*)calloc(1, sizeof(llama_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;
    m->n_layers = nl;
    m->n_heads = gf->n_heads;
    m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim;
    m->ffn = gf->ffn_dim;
    m->rope_base = gf->rope_freq_base;
    m->rms_eps = gf->rms_eps;

    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti >= 0) { m->q_dim = (int)gf->tensors[ti].shape[1]; m->head_dim = m->q_dim / m->n_heads; }
    else { m->head_dim = m->embed / m->n_heads; m->q_dim = m->n_heads * m->head_dim; }
    m->kv_dim = m->head_dim * m->n_kv_heads;

    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti >= 0) m->vocab = (int)gf->tensors[ti].shape[1];
    else if (gf->vocab_size > 0) m->vocab = gf->vocab_size;
    else m->vocab = 32000;

    printf("ocelli: E=%d H=%d KV=%d FFN=%d V=%d L=%d HD=%d Q=%d rope=%.0f rms=%.0e\n",
           m->embed, m->n_heads, m->n_kv_heads, m->ffn, m->vocab, nl, m->head_dim, m->q_dim,
           m->rope_base, m->rms_eps);

    m->tok_emb = load_wt(gf, "token_embd.weight");
    ti = gguf_find_tensor(gf, "output_norm.weight");  if (ti >= 0) m->out_norm = gguf_dequant(gf, ti);
    m->out_weight = load_wt(gf, "output.weight");
    m->has_output_weight = wt_ok(m->out_weight);

    for (int l = 0; l < nl; l++) {
        char name[128];
        #define L(field, fmt) do { snprintf(name, sizeof(name), fmt, l); \
            ti = gguf_find_tensor(gf, name); if (ti >= 0) m->layers[l].field = gguf_dequant(gf, ti); } while(0)
        #define LW(field, fmt) do { snprintf(name, sizeof(name), fmt, l); m->layers[l].field = load_wt(gf, name); } while(0)
        L(attn_norm, "blk.%d.attn_norm.weight");
        LW(wq, "blk.%d.attn_q.weight"); LW(wk, "blk.%d.attn_k.weight");
        LW(wv, "blk.%d.attn_v.weight"); LW(wo, "blk.%d.attn_output.weight");
        L(q_bias, "blk.%d.attn_q.bias"); L(k_bias, "blk.%d.attn_k.bias"); L(v_bias, "blk.%d.attn_v.bias");
        L(ffn_norm, "blk.%d.ffn_norm.weight");
        LW(wgate, "blk.%d.ffn_gate.weight"); LW(wup, "blk.%d.ffn_up.weight"); LW(wdown, "blk.%d.ffn_down.weight");
        #undef L
        #undef LW
    }
    if (!wt_ok(m->tok_emb) || !m->out_norm) { fprintf(stderr, "ocelli: missing critical weights\n"); return NULL; }
    if (!m->has_output_weight) printf("  (tied embeddings)\n");
    return m;
}

/* ── kv cache + forward (single token) ──────────────────────────────────────── */

typedef struct { float *k, *v; int max_seq, n_layers, kv_dim; } kv_cache;

static kv_cache* kv_new(int nl, int max_seq, int kv_dim) {
    kv_cache* kv = (kv_cache*)calloc(1, sizeof(kv_cache));
    kv->k = (float*)calloc((long)nl * max_seq * kv_dim, sizeof(float));
    kv->v = (float*)calloc((long)nl * max_seq * kv_dim, sizeof(float));
    kv->max_seq = max_seq; kv->n_layers = nl; kv->kv_dim = kv_dim;
    return kv;
}

static void llama_forward(llama_model* m, kv_cache* kv, int token, int pos, float* logits,
                          const float* emb_override) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads;
    int HD = m->head_dim, KVD = m->kv_dim, FFN = m->ffn, Q_DIM = m->q_dim;
    float eps = m->rms_eps; int gqa = H / KV;

    float *x = (float*)calloc(E, sizeof(float));
    // ── SPLICE POINT: image-placeholder positions get connector vision embeddings ──
    if (emb_override) memcpy(x, emb_override, E * sizeof(float));
    else if (m->tok_emb.f32) memcpy(x, m->tok_emb.f32 + (long)token * E, E * sizeof(float));
    else { const uint16_t *r = m->tok_emb.f16 + (long)token * E;   /* dequant one row */
           for (int i = 0; i < E; i++) x[i] = f16_to_f32_smol(r[i]); }

    float *xn = (float*)calloc(E, sizeof(float));
    float *q_all = (float*)calloc(Q_DIM, sizeof(float));
    float *k_new = (float*)calloc(KVD, sizeof(float));
    float *v_new = (float*)calloc(KVD, sizeof(float));
    float *attn_out = (float*)calloc(Q_DIM, sizeof(float));
    float *ffn_gate = (float*)calloc(FFN, sizeof(float));
    float *ffn_up = (float*)calloc(FFN, sizeof(float));
    float *ffn_out = (float*)calloc(E, sizeof(float));

    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(xn, x, m->layers[l].attn_norm, E, eps);
        mm_t_w(q_all, xn, m->layers[l].wq, 1, E, Q_DIM);
        mm_t_w(k_new, xn, m->layers[l].wk, 1, E, KVD);
        mm_t_w(v_new, xn, m->layers[l].wv, 1, E, KVD);
        add_bias(q_all, m->layers[l].q_bias, Q_DIM);
        add_bias(k_new, m->layers[l].k_bias, KVD);
        add_bias(v_new, m->layers[l].v_bias, KVD);
        for (int h = 0; h < H; h++)  rope(q_all + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++) rope(k_new + h*HD, pos, HD, m->rope_base);

        long base = (long)l * kv->max_seq * KVD;
        memcpy(kv->k + base + (long)pos * KVD, k_new, KVD * sizeof(float));
        memcpy(kv->v + base + (long)pos * KVD, v_new, KVD * sizeof(float));

        float scale = 1.0f / sqrtf((float)HD);
        memset(attn_out, 0, Q_DIM * sizeof(float));
        for (int h = 0; h < H; h++) {
            int kv_h = h / gqa;
            float *q = q_all + h * HD;
            float *scores = (float*)calloc(pos + 1, sizeof(float));
            for (int j = 0; j <= pos; j++) {
                float *kj = kv->k + base + (long)j * KVD + kv_h * HD;
                float dot = 0; for (int d = 0; d < HD; d++) dot += q[d] * kj[d];
                scores[j] = dot * scale;
            }
            softmax(scores, pos + 1);
            float *out_h = attn_out + h * HD;
            for (int j = 0; j <= pos; j++) {
                float *vj = kv->v + base + (long)j * KVD + kv_h * HD;
                for (int d = 0; d < HD; d++) out_h[d] += scores[j] * vj[d];
            }
            free(scores);
        }
        float *proj = (float*)calloc(E, sizeof(float));
        mm_t_w(proj, attn_out, m->layers[l].wo, 1, Q_DIM, E);
        for (int i = 0; i < E; i++) x[i] += proj[i];
        free(proj);

        rmsnorm(xn, x, m->layers[l].ffn_norm, E, eps);
        mm_t_w(ffn_gate, xn, m->layers[l].wgate, 1, E, FFN);
        mm_t_w(ffn_up, xn, m->layers[l].wup, 1, E, FFN);
        for (int i = 0; i < FFN; i++) { float g = ffn_gate[i]; ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i]; }
        mm_t_w(ffn_out, ffn_gate, m->layers[l].wdown, 1, FFN, E);
        for (int i = 0; i < E; i++) x[i] += ffn_out[i];
    }
    rmsnorm(xn, x, m->out_norm, E, eps);
    lmhead(m->has_output_weight ? m->out_weight : m->tok_emb, xn, logits, m->vocab, E);

    free(x); free(xn); free(q_all); free(k_new); free(v_new);
    free(attn_out); free(ffn_gate); free(ffn_up); free(ffn_out);
}

static int argmax(const float *x, int n) {
    int best = 0; float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

/* ── main ───────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <model.gguf> [--ids \"1 2 3\"] [--prompt \"text\"] [-n N]\n", argv[0]);
        printf("  --ids    feed raw token ids (isolates decoder; get them via llama-tokenize)\n");
        printf("  --prompt byte-level fallback (NOT real BPE — for quick smoke only)\n");
        return 1;
    }
    // PHASE 2: image preprocessing test (idefics3 tiling + normalize) — no model needed
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vision-test") && i + 1 < argc) {
            const char *ip = argv[i + 1];
            int nf = 0, S = 0;
            float *fr = smolvlm_preprocess(ip, &nf, &S);
            if (!fr) { fprintf(stderr, "preprocess failed: %s\n", ip); return 1; }
            long n = (long)nf * 3 * S * S;
            float mn = fr[0], mx = fr[0]; double sum = 0;
            for (long k = 0; k < n; k++) { float v = fr[k]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
            printf("vision-test: %s\n", ip);
            printf("  n_frames=%d  S=%d  shape/frame=[3,%d,%d]  total floats=%ld\n", nf, S, S, S, n);
            printf("  value range: min=%.4f max=%.4f mean=%.4f  (expect [-1,1], mean~0)\n", mn, mx, sum / n);
            printf("  image tokens (64/frame) = %d\n", nf * 64);
            free(fr);
            return 0;
        }
    }

    // SigLIP vision-tower probe — usage: ocelli x --siglip-test <image> <mmproj.gguf>
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--siglip-test") && i + 2 < argc) {
            const char *ip = argv[i + 1], *mmp = argv[i + 2];
            siglip_model *vm = siglip_load(mmp);
            if (!vm) { fprintf(stderr, "siglip_load failed: %s\n", mmp); return 1; }
            int nf = 0, S = 0;
            float *fr = smolvlm_preprocess(ip, &nf, &S);
            if (!fr) { fprintf(stderr, "preprocess failed: %s\n", ip); return 1; }
            int P = siglip_n_patches(vm), D = siglip_hidden(vm);
            float *h = (float*)malloc((long)P * D * sizeof(float));
            if (!h || siglip_encode(vm, fr, h) != 0) { fprintf(stderr, "siglip_encode failed\n"); return 1; }
            long n = (long)P * D, nan = 0; float mn = h[0], mx = h[0]; double sum = 0, csum = 0;
            for (long t = 0; t < n; t++) { float x = h[t];
                if (x != x) nan++; if (x < mn) mn = x; if (x > mx) mx = x; sum += x; csum += (double)x * (t % 97 + 1); }
            double tok0 = 0; for (int j = 0; j < D; j++) tok0 += h[j]; tok0 /= D;
            printf("siglip-test: image=%s mmproj=%s\n", ip, mmp);
            printf("  vision hidden states = [%d, %d]  (frame 0 of %d)\n", P, D, nf);
            printf("  NaN=%ld  min=%.4f  max=%.4f  mean=%.5f  token0_mean=%.5f\n", nan, mn, mx, sum / n, tok0);
            printf("  checksum=%.6f  (determinism: must be identical across runs)\n", csum);
            /* PHASE 4: pixel-shuffle connector -> visual embeddings in text dim */
            int NV = siglip_n_vis_tokens(vm), TD = siglip_text_dim(vm);
            float *vemb = (float*)malloc((long)NV * TD * sizeof(float));
            if (vemb && siglip_connect(vm, h, vemb) == 0) {
                long vn = (long)NV * TD, vnan = 0; float vmn = vemb[0], vmx = vemb[0]; double vsum = 0, vcs = 0;
                for (long t = 0; t < vn; t++) { float x = vemb[t];
                    if (x != x) vnan++; if (x < vmn) vmn = x; if (x > vmx) vmx = x; vsum += x; vcs += (double)x * (t % 97 + 1); }
                printf("  connector: visual embeddings = [%d, %d]  NaN=%ld min=%.4f max=%.4f mean=%.5f\n",
                       NV, TD, vnan, vmn, vmx, vsum / vn);
                printf("  connector checksum=%.6f\n", vcs);
            } else { fprintf(stderr, "siglip_connect failed\n"); return 1; }
            free(vemb); free(h); free(fr); siglip_free(vm);
            return 0;
        }
    }

    const char *model_path = argv[1];
    const char *ids_str = NULL, *prompt = NULL, *text = NULL, *image_path = NULL, *mmproj_path = NULL;
    int max_tokens = 16;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i+1 < argc) ids_str = argv[++i];
        else if (!strcmp(argv[i], "--text") && i+1 < argc) text = argv[++i];
        else if ((!strcmp(argv[i], "--prompt") || !strcmp(argv[i], "-p")) && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--image") && i+1 < argc) image_path = argv[++i];
        else if (!strcmp(argv[i], "--mmproj") && i+1 < argc) mmproj_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) max_tokens = atoi(argv[++i]);
    }

    gguf_file* gf = gguf_open(model_path);
    if (!gf) return 1;
    llama_model* model = llama_load(gf);
    gguf_close(gf); gf = NULL;   // weights copied into model (f16/f32); free ~313MB raw GGUF data now
    if (!model) return 1;

    // GPT-2 byte-level BPE over the GGUF tokenizer (notorch examples/bpe.c, vendored)
    bpe_tokenizer *bpe = bpe_load(model_path);
    if (bpe) printf("bpe: %d tokens loaded\n", bpe_n_vocab(bpe));
    else printf("bpe: load failed (id/byte modes only)\n");

    // ── PHASE 5: image -> text (vision tower + connector + splice, end-to-end) ──
    if (image_path && mmproj_path && bpe) {
        siglip_model *vm = siglip_load(mmproj_path);
        if (!vm) { fprintf(stderr, "mmproj load failed: %s\n", mmproj_path); return 1; }
        int nf = 0, S = 0, n_rows = 1, n_cols = 1;
        float *fr = smolvlm_preprocess_grid(image_path, &nf, &S, &n_rows, &n_cols);
        if (!fr) { fprintf(stderr, "preprocess failed: %s\n", image_path); return 1; }
        int P = siglip_n_patches(vm), NV = siglip_n_vis_tokens(vm), TD = siglip_text_dim(vm);
        if (TD != model->embed) { fprintf(stderr, "dim mismatch vis=%d text=%d\n", TD, model->embed); return 1; }

        /* Every frame goes through the tower: a 1024x1024 photo is 16 tiles plus
         * one global frame, and encoding only the first one is what made small
         * text unreadable — the engine was looking at a 17th of the detail. */
        long frame_floats = (long)3 * S * S;
        int n_vis_total = nf * NV;
        printf("image=%s  frames=%d (%dx%d tiles + global), %d visual tokens\n",
               image_path, nf, n_rows, n_cols, n_vis_total);
        fflush(stdout);
        float *hid  = (float*)malloc((long)P * siglip_hidden(vm) * sizeof(float));
        float *vemb = (float*)malloc((long)n_vis_total * TD * sizeof(float));
        if (!hid || !vemb) { fprintf(stderr, "vision alloc failed\n"); return 1; }
        double t_vis = now_ms();
        for (int f = 0; f < nf; f++) {
            if (siglip_encode(vm, fr + (long)f * frame_floats, hid) != 0 ||
                siglip_connect(vm, hid, vemb + (long)f * NV * TD) != 0) {
                fprintf(stderr, "vision encode/connect failed on frame %d/%d\n", f, nf); return 1; }
            printf("\r  vision: frame %d/%d", f + 1, nf); fflush(stdout);
        }
        printf("\r  vision: %d frames in %.0f ms            \n", nf, now_ms() - t_vis);
        free(hid); free(fr);

        /* idefics3 layout. Split form (tiles present) is row-major per-tile
         * markers, a newline after each tile row, then the global frame — the
         * token count of that layout reproduces the reference runtime's prompt
         * length on the same image (1143 tokens for a 4x4 grid). Single-frame
         * form is the 84-token layout verified earlier against the same oracle. */
        const char *instr = prompt ? prompt : "Describe this image in one sentence.";
        size_t cap = (size_t)n_vis_total * 8 + (size_t)nf * 32 + strlen(instr) + 512;
        char *buf = (char*)malloc(cap);
        if (!buf) { fprintf(stderr, "prompt alloc failed\n"); return 1; }
        size_t off = (size_t)snprintf(buf, cap, "<|im_start|>User:\n");
        if (nf > 1) {
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    off += snprintf(buf + off, cap - off,
                                    "<fake_token_around_image><row_%d_col_%d>", r + 1, c + 1);
                    for (int j = 0; j < NV; j++) off += snprintf(buf + off, cap - off, "<image>");
                }
                off += snprintf(buf + off, cap - off, "\n");
            }
            off += snprintf(buf + off, cap - off, "\n<fake_token_around_image><global-img>");
            for (int j = 0; j < NV; j++) off += snprintf(buf + off, cap - off, "<image>");
            snprintf(buf + off, cap - off,
                     "<fake_token_around_image>%s<end_of_utterance>\nAssistant:", instr);
        } else {
            off += snprintf(buf + off, cap - off, "<fake_token_around_image><global-img>");
            for (int j = 0; j < NV; j++) off += snprintf(buf + off, cap - off, "<image>");
            snprintf(buf + off, cap - off,
                     "<fake_token_around_image>\n%s<end_of_utterance>\nAssistant:", instr);
        }

        /* 17 frames x 64 visual tokens plus chat wrapping needs room to breathe */
        int img_max = 64, MS = 2048;
        int *toks = (int*)calloc(MS, sizeof(int));
        int n_tok = bpe_encode(bpe, buf, toks, MS - img_max);
        free(buf);
        printf("image=%s  prompt=%d tok (vis=%d, frames=%d = %dx%d tiles + global)\n",
               image_path, n_tok, n_vis_total, nf, n_rows, n_cols);
        double t_start = now_ms();

        kv_cache *kv = kv_new(model->n_layers, MS, model->kv_dim);
        float *logits = (float*)calloc(model->vocab, sizeof(float));
        int vis_slot = 0;
        for (int i = 0; i < n_tok; i++) {
            const float *ov = NULL;
            if (toks[i] == 49190 && vis_slot < n_vis_total) ov = vemb + (long)(vis_slot++) * TD;  // splice
            llama_forward(model, kv, toks[i], i, logits, ov);
        }
        if (vis_slot != n_vis_total)   /* layout and budget must agree, or the tail is blind */
            fprintf(stderr, "warning: spliced %d of %d visual tokens\n", vis_slot, n_vis_total);
        double t_prompt = now_ms() - t_start;
        char piece[256];
        int n_gen = 0;
        printf("OURS: \"");
        for (int step = 0; step < img_max; step++) {
            int next = argmax(logits, model->vocab);
            if (next == 2 || next == 49279) break;   // eos / <end_of_utterance>
            piece[0] = 0; bpe_decode_token(bpe, next, piece, sizeof(piece));
            printf("%s", piece); fflush(stdout);
            n_gen++;
            int pos = n_tok + step;
            if (pos >= MS - 1) break;
            llama_forward(model, kv, next, pos, logits, NULL);
        }
        printf("\"\n");
        /* the eye reports its own cost. Prompt and generation are timed apart:
         * one number over both would call 84 prompt forwards a generation rate.
         * ru_maxrss is bytes on macOS. */
        double t_gen = now_ms() - t_start - t_prompt;
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        double rss_mb = (double)ru.ru_maxrss / (1024.0 * 1024.0);
        printf("-- prompt %d tok in %.0f ms (%.1f tok/s) | gen %d tok in %.0f ms (%.1f tok/s) | peak RSS %.0f MB --\n",
               n_tok, t_prompt, n_tok / (t_prompt / 1000.0),
               n_gen, t_gen, n_gen > 0 ? n_gen / (t_gen / 1000.0) : 0.0, rss_mb);
        free(vemb); free(toks); free(logits); siglip_free(vm);
        bpe_free(bpe);   // gf already closed after llama_load
        return 0;
    }

    // build prompt token ids
    int max_seq = 1024;
    int *tokens = (int*)calloc(max_seq, sizeof(int));
    int n_tok = 0;
    if (ids_str) {
        char *buf = strdup(ids_str), *tok = strtok(buf, " ,");
        while (tok && n_tok < max_seq - max_tokens) { tokens[n_tok++] = atoi(tok); tok = strtok(NULL, " ,"); }
        free(buf);
    } else if (text && bpe) {
        n_tok = bpe_encode(bpe, text, tokens, max_seq - max_tokens);   // real GPT-2 BPE
    } else {
        tokens[n_tok++] = 1; // BOS
        const char *p = prompt ? prompt : "Hello";
        for (int i = 0; p[i] && n_tok < max_seq - max_tokens; i++) tokens[n_tok++] = (unsigned char)p[i];
    }

    printf("prompt: %d tokens [", n_tok);
    for (int i = 0; i < n_tok; i++) printf("%d%s", tokens[i], i+1<n_tok?" ":"");
    printf("]\n");

    kv_cache* kv = kv_new(model->n_layers, max_seq, model->kv_dim);
    float *logits = (float*)calloc(model->vocab, sizeof(float));

    double t0 = now_ms();
    for (int i = 0; i < n_tok; i++) llama_forward(model, kv, tokens[i], i, logits, NULL);

    printf("\n-- greedy decode (%d tokens) --\n", max_tokens);
    char piece[256], out_text[4096]; out_text[0] = 0; int out_len = 0;
    for (int step = 0; step < max_tokens; step++) {
        int next = argmax(logits, model->vocab);
        piece[0] = 0;
        if (bpe) bpe_decode_token(bpe, next, piece, sizeof(piece));
        printf("  [%d] id=%d  tok=\"%s\"\n", step, next, piece);
        if (next == 2) { printf("  (EOS)\n"); break; }   // SmolLM2 eos=2
        int pl = (int)strlen(piece);
        if (out_len + pl < (int)sizeof(out_text) - 1) { strcpy(out_text + out_len, piece); out_len += pl; }
        int pos = n_tok + step;
        if (pos >= max_seq - 1) break;
        llama_forward(model, kv, next, pos, logits, NULL);
    }
    printf("\ngenerated: \"%s\"\n", out_text);
    printf("-- %.0f ms --\n", now_ms() - t0);

    free(logits); free(tokens);
    if (bpe) bpe_free(bpe);
    // gf already closed after llama_load
    return 0;
}
