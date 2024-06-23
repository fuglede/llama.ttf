// Adapted from https://github.com/gaxler/llama2.rs/
//
// MIT License
//
// Copyright (c) 2023 Andrej
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std::mem;

static STORIES: &[u8; 60816028] = include_bytes!("stories15M.bin"); // Yup ...
static TOKENIZER: &[u8; 304713] = include_bytes!("tokenizer.bin");

const CONF_VALS: usize = 7;
const CONF_SIZE: usize = std::mem::size_of::<[i32; CONF_VALS]>();

#[derive(Debug, Clone, Copy)]
pub struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    pub vocab_size: usize,
    seq_len: usize,
    shared_weights: bool,
}

/// Exexute LLama step
pub trait LamaExecuter<Buffer> {
    fn step(&mut self, token: usize, pos: usize, cfg: &Config, state: &mut ExecutionState<Buffer>);
}

/// Executte Llama layer
pub trait LLamaLayer<Buffer> {
    /// RMS norm residual stream and get Q,K,V matrices
    fn rms_and_qkv(&self, cfg: &Config, state: &mut ExecutionState<Buffer>);
    /// Rotate q and k heads according to position in seq (RoPE)
    fn rope(
        &self,
        pos: usize,
        cfg: &Config,
        state: &mut ExecutionState<Buffer>,
        rope_imag: &Buffer,
        rope_real: &Buffer,
    );
    /// Cache sequence of Q, K (to be used for attention computation)
    fn cache_kv(&mut self, pos: usize, cfg: &Config, state: &ExecutionState<Buffer>);
    /// (per head) Calculate Attention weights, accumulate value according to weights
    fn attention(&self, pos: usize, cfg: &Config, state: &ExecutionState<Buffer>);
    /// Merge all heads and add result to residula stream
    fn merge_heads_to_resid_stream(&self, state: &mut ExecutionState<Buffer>);
    /// RMS norm residual stream,
    /// apply FeedForward to normalized
    /// add to residual stream
    fn ffn(&self, state: &mut ExecutionState<Buffer>);
}

pub trait LinearWeight<T> {
    fn mat_vec(&self, vec: &T, dst: &mut T);
}
pub trait RMSNormWeight<T> {
    fn rms_norm(&self, vec: &T, out: &mut T);
    fn inplace_rms_norm(&self, vec: &mut T);
}

pub trait EmbeddingTable<Buf>: LinearWeight<Buf> {
    fn token_to_resid_stream(&self, token: usize, dst: &mut Buf, cfg: &Config);
}

pub struct LlamaWeights<Layer, Rms, Emb, Buf> {
    /// (vocab_size, dim)
    pub embeddings: Emb,
    pub layers: Vec<Layer>,
    /// (dim,)
    pub rms_final: Rms,
    /// (seq_len, head_size/2)
    pub rope_real: Buf,
    /// (seq_len, head_size/2)
    pub rope_imag: Buf,
    pub wcls: Option<Emb>,
}

pub struct LayerWeights<Lin, Rms, Buf> {
    rms_attn: Rms,
    rms_ffn: Rms,
    wq: Lin,
    wk: Lin,
    wv: Lin,
    wo: Lin,
    w1: Lin,
    w2: Lin,
    w3: Lin,
    /// (seq_len, dim)
    k_cache: Buf,
    /// (seq_len, dim)
    v_cache: Buf,
}

type Ty = f32;
type CPULayerFloat = LayerWeights<Vec<Ty>, Vec<Ty>, Vec<Ty>>;
type Llama2CPUFloat = LlamaWeights<CPULayerFloat, Vec<Ty>, Vec<Ty>, Vec<Ty>>;

pub struct ExecutionState<Buffer> {
    /// Shape:(dim,)
    x: Buffer,
    /// Shape:(dim,)
    xb: Buffer,
    /// Shape:(dim,)
    xb2: Buffer,
    /// Shape:(hidden_dim,)
    h1: Buffer,
    /// Shape:(hidden_dim,)
    h2: Buffer,
    /// (dim,): Q, buffers
    q: Buffer,
    /// (dim,): K buffer
    k: Buffer,
    /// (dim,): V buffer
    v: Buffer,
    /// (n_heads, seq_len): Attention Weight Buffer
    att: Buffer,
    /// Logits: (vocab_size, )
    pub logits: Buffer,
}

// f32 CPU implementation of Llama2
impl<L, Rms, Emb> LamaExecuter<Vec<Ty>> for LlamaWeights<L, Rms, Emb, Vec<Ty>>
where
    L: LLamaLayer<Vec<Ty>>,
    Rms: RMSNormWeight<Vec<Ty>>,
    Emb: EmbeddingTable<Vec<Ty>>,
{
    fn step(
        &mut self,
        token: usize,
        pos: usize,
        cfg: &Config,
        state: &mut ExecutionState<Vec<Ty>>,
    ) {
        // copy token embedding to residual stream
        self.embeddings
            .token_to_resid_stream(token, &mut state.x, cfg);

        for ld in self.layers.iter_mut() {
            ld.rms_and_qkv(cfg, state);
            ld.rope(pos, cfg, state, &self.rope_imag, &self.rope_real);
            ld.cache_kv(pos, cfg, state);
            ld.attention(pos, cfg, state);
            ld.merge_heads_to_resid_stream(state);
            ld.ffn(state);
        }

        self.rms_final.inplace_rms_norm(&mut state.x);

        if self.wcls.is_none() {
            self.embeddings.mat_vec(&state.x, &mut state.logits);
        } else {
            let w = self.wcls.as_ref().unwrap();
            w.mat_vec(&state.x, &mut state.logits);
        }
    }
}

// f32 Implementation of Llama2 layer
impl<Lin, Rms> LLamaLayer<Vec<Ty>> for LayerWeights<Lin, Rms, Vec<Ty>>
where
    Lin: LinearWeight<Vec<Ty>>,
    Rms: RMSNormWeight<Vec<Ty>>,
{
    fn rms_and_qkv(&self, cfg: &Config, state: &mut ExecutionState<Vec<Ty>>) {
        self.rms_attn.rms_norm(&state.x, &mut state.xb);
        self.wq.mat_vec(&state.xb, &mut state.q);
        self.wk.mat_vec(&state.xb, &mut state.k);
        self.wv.mat_vec(&state.xb, &mut state.v);
    }
    fn rope(
        &self,
        pos: usize,
        cfg: &Config,
        state: &mut ExecutionState<Vec<Ty>>,
        rope_imag: &Vec<Ty>,
        rope_real: &Vec<Ty>,
    ) {
        let head_size = cfg.dim / cfg.n_heads;

        let q_heads = state.q.chunks_exact_mut(head_size);
        let k_heads = state.k.chunks_exact_mut(head_size);

        for (q, k) in q_heads.zip(k_heads) {
            let mut re = rope_real[pos * head_size / 2..].iter().take(head_size / 2);

            let mut im = rope_imag[pos * head_size / 2..].iter().take(head_size / 2);

            for (qq, kk) in q.chunks_exact_mut(2).zip(k.chunks_exact_mut(2)) {
                let (q0, q1) = (qq[0], qq[1]);
                let (k0, k1) = (kk[0], kk[1]);
                let fcr = re.next().unwrap();
                let fci = im.next().unwrap();
                qq[0] = q0 * fcr - q1 * fci;
                qq[1] = q0 * fci + q1 * fcr;
                kk[0] = k0 * fcr - k1 * fci;
                kk[1] = k0 * fci + k1 * fcr;
            }
        }
    }
    fn cache_kv(&mut self, pos: usize, cfg: &Config, state: &ExecutionState<Vec<Ty>>) {
        let dst_k = &mut self.k_cache[pos * cfg.dim..(pos + 1) * cfg.dim];
        let dst_v = &mut self.v_cache[pos * cfg.dim..(pos + 1) * cfg.dim];
        dst_k.copy_from_slice(&state.k);
        dst_v.copy_from_slice(&state.v);
    }

    fn attention(&self, pos: usize, cfg: &Config, state: &ExecutionState<Vec<Ty>>) {
        // State is a shared reference becasue we will pass that to multiple threads.
        // However we are going to take an unsafe mutable references inside the threads
        // We can do that because each thread handles a single head and head data is disjoint
        let head_size = cfg.dim / cfg.n_heads;
        let k_cache = self.k_cache.as_slice();
        let v_cache = self.v_cache.as_slice();

        let attn_lambda = |h: usize| {
            let q = unsafe { _uncheked_slice(&state.q, h * head_size, head_size) };
            // head attention weights of len (seq_len, )
            let att_weights =
                unsafe { _uncheked_mut_slice(&state.att, h * cfg.seq_len, cfg.seq_len) };
            let xb = unsafe { _uncheked_mut_slice(&state.xb, h * head_size, head_size) };
            // head K cache of (seq_len,head_size)
            let mut head_k_cache = k_cache.chunks_exact(head_size).skip(h).step_by(cfg.n_heads);

            // do <Q,K> for head
            for t in 0..=pos {
                let k = head_k_cache.next().unwrap(); // head_size
                let score = k
                    .iter()
                    .zip(q.iter())
                    .fold(0 as Ty, |acc, (_k, _q)| acc + _k * _q);
                let score = score / (head_size as Ty).sqrt();
                unsafe {
                    *att_weights.get_unchecked_mut(t) = score;
                }
            }

            // head V cache of (seq_len, head_size)
            let head_v_cache = v_cache.chunks_exact(head_size).skip(h).step_by(cfg.n_heads);
            inplace_softmax(&mut att_weights[..=pos]);
            // reset buffer head out buffer
            xb.iter_mut().for_each(|v| *v = 0 as Ty);
            // accumulate cached values to current buffer
            // according to attention prob. (normalized weights)
            for (vals, p_attn) in head_v_cache.zip(att_weights.iter()).take(pos + 1) {
                vals.iter()
                    .zip(xb.iter_mut())
                    .for_each(|(v, dst)| *dst += v * p_attn)
            }
        };

        #[cfg(feature = "parallel")]
        (0..cfg.n_heads).into_par_iter().for_each(attn_lambda);

        #[cfg(not(feature = "parallel"))]
        (0..cfg.n_heads).for_each(attn_lambda);
    }

    fn merge_heads_to_resid_stream(&self, state: &mut ExecutionState<Vec<Ty>>) {
        // merge heads
        // at this point result of all heads in in x[1],
        // Linearly  merge all heads into a new buffer x[2]
        self.wo.mat_vec(&state.xb, &mut state.xb2);

        // add attention result to  residual stream
        state
            .x
            .iter_mut()
            .zip(state.xb2.iter())
            .for_each(|(x, xb)| *x += *xb);
    }

    fn ffn(&self, state: &mut ExecutionState<Vec<Ty>>) {
        // normalize residual stream before FFN
        self.rms_ffn.rms_norm(&state.x, &mut state.xb);

        // FFN:
        //  z = SiLU(W1 \dot x) * (W3 \dot x)
        // out = (W2 \dot z)
        self.w1.mat_vec(&state.xb, &mut state.h1);
        self.w3.mat_vec(&state.xb, &mut state.h2);

        // silu hidden
        for h1 in state.h1.iter_mut() {
            // 1 / 1 + exp(-hv)
            let _scaler = (1 as Ty) / ((1 as Ty) + (-*h1).exp());
            *h1 = *h1 * _scaler;
        }

        // combine hidden state with multiplication
        for (h1, &h2) in state.h1.iter_mut().zip(state.h2.iter()) {
            *h1 *= h2;
        }
        self.w2.mat_vec(&state.h1, &mut state.xb);

        // add FFN result to residual stream
        state
            .x
            .iter_mut()
            .zip(state.xb.iter())
            .for_each(|(x, z)| *x += *z);
    }
}

/// Helper to simplifiy buffer init
pub trait DefualtBuffer {
    fn zeros(size: usize) -> Self;
}

impl DefualtBuffer for Vec<Ty> {
    fn zeros(size: usize) -> Self {
        vec![0 as Ty; size]
    }
}

impl<T: DefualtBuffer> ExecutionState<T> {
    pub fn init(cfg: &Config) -> Self {
        Self {
            x: T::zeros(cfg.dim),
            xb: T::zeros(cfg.dim),
            xb2: T::zeros(cfg.dim),
            h1: T::zeros(cfg.hidden_dim),
            h2: T::zeros(cfg.hidden_dim),
            q: T::zeros(cfg.dim),
            k: T::zeros(cfg.dim),
            v: T::zeros(cfg.dim),
            att: T::zeros(cfg.n_heads * cfg.seq_len),
            logits: T::zeros(cfg.vocab_size),
        }
    }
}

impl EmbeddingTable<Vec<Ty>> for Vec<Ty> {
    fn token_to_resid_stream(&self, pos: usize, dst: &mut Vec<Ty>, cfg: &Config) {
        let dim = dst.len();
        self.chunks_exact(dim)
            .skip(pos)
            .take(1)
            .for_each(|src| dst.as_mut_slice().copy_from_slice(src));
    }
}

impl LinearWeight<Vec<Ty>> for Vec<Ty> {
    fn mat_vec(&self, vec: &Vec<Ty>, dst: &mut Vec<Ty>) {
        matmul(dst, vec, &self); // in_dim is infered form x. need to remove from function sig
    }
}

#[inline]
fn _norm_const(vec: &[Ty]) -> Ty {
    let dim = vec.len() as Ty;
    let ssq = vec.iter().fold(0f32, |init, &v| init + v * v) / dim;
    (1 as Ty) / (ssq + 1e-5).sqrt()
}

impl RMSNormWeight<Vec<Ty>> for Vec<Ty> {
    fn rms_norm(&self, vec: &Vec<Ty>, out: &mut Vec<Ty>) {
        let inv_denom = _norm_const(vec);

        let w_it = self.iter();
        let normed = vec.iter().zip(w_it).map(|(xx, ww)| xx * ww * inv_denom);
        out.iter_mut().zip(normed).for_each(|(dst, src)| *dst = src);
    }

    fn inplace_rms_norm(&self, vec: &mut Vec<Ty>) {
        let inv_denom = _norm_const(vec);

        let w_it = self.iter();
        vec.iter_mut()
            .zip(w_it)
            .for_each(|(dst, w)| (*dst) *= inv_denom * w);
    }
}

fn _alloc_and_read(offset: usize, size: usize) -> Vec<Ty> {
    unsafe {
        let float_ptr = STORIES.as_ptr() as *const Ty;
        let data = std::slice::from_raw_parts(float_ptr.add(offset), size);
        data.to_vec()
    }
}

/// Load raw weights from Karphaty's models
fn load_raw_karphaty(cfg: &Config) -> (Vec<Vec<Ty>>, Option<Vec<Ty>>) {
    let f = |offset: usize, s: usize| _alloc_and_read(offset, s);
    let head_size = cfg.dim / cfg.n_heads;

    let sizes = vec![
        cfg.vocab_size * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.dim,
        cfg.n_layers * cfg.dim,
        cfg.n_layers * cfg.dim * cfg.hidden_dim,
        cfg.n_layers * cfg.dim * cfg.hidden_dim,
        cfg.n_layers * cfg.dim * cfg.hidden_dim,
        cfg.dim,
        cfg.seq_len * (head_size / 2),
        cfg.seq_len * (head_size / 2),
    ];
    let mut offset = 7;
    let mut parameters: Vec<Vec<Ty>> = vec![];
    for size in sizes {
        parameters.push(f(offset, size));
        offset += size;
    }

    (
        parameters,
        cfg.shared_weights
            .then(|| f(offset, cfg.vocab_size * cfg.dim)),
    )
}

impl Llama2CPUFloat {
    pub fn load_weights(cfg: &Config) -> Self {
        let (weights, wcls) = load_raw_karphaty(cfg);
        let embeddings = weights[0].clone();

        // Go over all layered weights, and make layer chunk out of them
        let mut w_layer_iters = weights[1..10]
            .iter()
            .map(|v| {
                let csize = v.len() / cfg.n_layers;
                v.chunks(csize).map(|l| l.to_vec())
            })
            .collect::<Vec<_>>();

        let layers = (0..cfg.n_layers)
            .map(|l| LayerWeights::<Vec<Ty>, Vec<Ty>, Vec<Ty>> {
                rms_attn: w_layer_iters[0].next().unwrap(),
                wq: w_layer_iters[1].next().unwrap(),
                wk: w_layer_iters[2].next().unwrap(),
                wv: w_layer_iters[3].next().unwrap(),
                wo: w_layer_iters[4].next().unwrap(),
                rms_ffn: w_layer_iters[5].next().unwrap(),
                w1: w_layer_iters[6].next().unwrap(),
                w2: w_layer_iters[7].next().unwrap(),
                w3: w_layer_iters[8].next().unwrap(),
                k_cache: vec![0 as Ty; cfg.seq_len * cfg.dim],
                v_cache: vec![0 as Ty; cfg.seq_len * cfg.dim],
            })
            .collect();

        let rms_final = weights[10].clone();
        let rope_real = weights[11].clone();
        let rope_imag = weights[12].clone();

        Self {
            embeddings,
            layers,
            rms_final,
            rope_real,
            rope_imag,
            wcls,
        }
    }
}

impl Config {
    /// Read raw bytes and force those to be our config type (which conforms to C mem layout)
    pub fn from_file() -> Self {
        let mut buffer = [0; CONF_SIZE];
        for j in 0..28 {
            buffer[j] = STORIES[j];
        }
        //model_bin.read_exact(&mut buffer).unwrap();
        let raw_conf = unsafe { mem::transmute::<[u8; CONF_SIZE], [i32; CONF_VALS]>(buffer) };
        let (vocab_size, shared_weights) = if raw_conf[5] < 0 {
            (-raw_conf[5] as usize, true)
        } else {
            (raw_conf[5] as usize, false)
        };

        Self {
            dim: raw_conf[0] as usize,
            hidden_dim: raw_conf[1] as usize,
            n_layers: raw_conf[2] as usize,
            n_heads: raw_conf[3] as usize,
            n_kv_heads: raw_conf[4] as usize,
            vocab_size,
            seq_len: raw_conf[6] as usize,
            shared_weights,
        }
    }
}

pub struct Vocab {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl Vocab {
    pub fn from_file(vocab_size: usize) -> Self {
        let mut bytes = Vec::<u8>::new();
        let mut offsets = vec![0usize; 1];
        let vocab_bytes = TOKENIZER;
        let mut pointer = 0;
        let mut len = [0; 4];
        let mut val = [0; 1];
        for vs in 0..vocab_size {
            for j in 0..4 {
                len[j] = vocab_bytes[pointer + j];
            }
            pointer += 4;
            //vocab_bin.read_exact(&mut len).unwrap();
            let l = unsafe { mem::transmute::<[u8; 4], i32>(len) };
            offsets.push(offsets.last().unwrap() + l as usize);
            (0..l).for_each(|_| {
                val[0] = vocab_bytes[pointer];
                pointer += 1;
                // vocab_bin.read_exact(&mut val).unwrap();
                bytes.extend(val);
            });
        }

        assert_eq!(offsets.len(), vocab_size + 1);

        Self { bytes, offsets }
    }

    pub fn get_token(&self, idx: usize) -> &str {
        let (st, en) = (self.offsets[idx], self.offsets[idx + 1]);
        let b = &self.bytes[st..en];
        std::str::from_utf8(b).unwrap()
    }
}

#[cfg(not(feature = "parallel"))]
fn matmul(out: &mut [Ty], x: &[Ty], w: &[Ty]) {
    let stride = x.len();
    for (row, out_elem) in w.chunks_exact(stride).zip(out.iter_mut()) {
        let val = row
            .iter()
            .zip(x.iter())
            .fold(0 as Ty, |acc, (&_w, &_x)| acc + _w * _x);
        *out_elem = val;
    }
}

/// We can safely borrow disjoint parts of slices, but its really hard for the borrow checker to know that this is safe
unsafe fn _uncheked_mut_slice(s: &[Ty], offset: usize, size: usize) -> &mut [Ty] {
    let ptr: *mut f32 = s.as_ptr() as *mut Ty;
    let st = ptr.add(offset);
    std::slice::from_raw_parts_mut(st, size)
}

/// We can safely borrow disjoint parts of slices, but its really hard for the borrow checker to know that this is safe
unsafe fn _uncheked_slice<Q>(s: &[Q], offset: usize, size: usize) -> &[Q] {
    let ptr = s.as_ptr();
    let st = ptr.add(offset);
    std::slice::from_raw_parts(st, size)
}

fn inplace_softmax(x: &mut [Ty]) {
    let max_val = x.iter().fold(Ty::NAN, |acc, &v| v.max(acc));
    let mut denom = 0 as Ty;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        denom += *v;
    }

    x.iter_mut().for_each(|v| *v /= denom);
}
