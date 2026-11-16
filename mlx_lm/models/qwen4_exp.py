# Copyright © 2026 Apple Inc.

# Qwen3.8-Flash-Next: qwen3_next with QSA sparse attention, hyper-connections
# and an n-gram embedding (PLE).

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, CacheList, KVCache
from .gated_delta import gated_delta_update
from .switch_layers import SwitchGLU


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    full_attention_interval: int = 4
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    # gated deltanet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # QSA
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # n-gram / PLE
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    ple_embed_dim: int = 2560
    ple_layer_ids: list = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    # Not in config.json; the n-gram hash depends on it.
    seed: int = 1234
    eos_token_id: Any = 248044
    partial_rotary_factor: float = 0.25
    rope_parameters: dict = field(default_factory=dict)
    rope_theta: float = 10_000_000.0
    tie_word_embeddings: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
        rp = self.text.rope_parameters or {}
        self.text.rope_theta = float(rp.get("rope_theta", self.text.rope_theta))
        self.text.partial_rotary_factor = float(
            rp.get("partial_rotary_factor", self.text.partial_rotary_factor)
        )
        if not self.text.layer_types:
            n, k = self.text.num_hidden_layers, self.text.full_attention_interval
            self.text.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(n)
            ]


class RMSNorm(nn.Module):
    """RMSNorm with one statistic per `group_size` slice, e.g. per stream of the
    hyper-connections. `sanitize` folds the checkpoint's `1 + w` into `weight`."""

    def __init__(self, dim: int, group_size: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"dim {dim} is not divisible by group_size {group_size}")

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        x = mx.fast.rms_norm(x, None, self.eps).reshape(shape)
        return x * self.weight


class RMSNormGated(nn.Module):
    """RMSNorm with an optional gate; the checkpoint stores `weight` as is."""

    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        out = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return out.astype(x.dtype)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        g = act(gate.astype(mx.float32))
        return (g * out.astype(mx.float32)).astype(x.dtype)


def _positions(offset: Union[int, mx.array], S: int) -> mx.array:
    """Positions of `S` tokens from `offset`: (1, S) for an int offset, (B, S)
    for one offset per row."""
    if isinstance(offset, mx.array):
        return offset.reshape(-1, 1) + mx.arange(S)
    return mx.arange(offset, offset + S)[None]


def _left_padding(cache) -> Optional[mx.array]:
    """Left padding of a batched cache, per row, or None if the rows align."""
    pad = getattr(cache, "left_padding", None)
    if pad is None or pad.max().item() == 0:
        return None
    return pad


def _keep_last(x: mx.array, n: int, lengths: Optional[mx.array]) -> mx.array:
    """The last `n` steps of `x`; a right padded row ends at its length."""
    if lengths is None:
        # A copy: a slice would keep all of `x` alive in the cache.
        return mx.contiguous(x[:, -n:])
    positions = mx.clip(lengths, 0, x.shape[1] - n)[:, None] + mx.arange(n)
    if x.ndim == 3:
        positions = positions[..., None]
    return mx.take_along_axis(x, positions, axis=1)


class QSAIndexer(nn.Module):
    """Select, per query, a budget of compressed key blocks."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.kv_heads = args.indexer_kv_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope = nn.RoPE(rotary_dim, base=args.rope_theta)
        # Block n starts at position n * compress_ratio.
        self.block_rope = nn.RoPE(
            rotary_dim, base=args.rope_theta, scale=self.compress_ratio
        )

    def __call__(self, x, cache, offset, left_padding=None) -> Optional[mx.array]:
        B, S, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)

        if cache is not None:
            # As in deepseek_v32: a KVCache for the keys, zeros for the values.
            raw_k, _ = cache.update_and_fetch(raw_k[:, None], raw_k[:, None])
            cache.values = mx.zeros_like(cache.keys)
            raw_k = raw_k[:, 0]
        kv_len = raw_k.shape[1]

        # Every visible token fits in the budget: the causal mask is enough.
        if kv_len <= self.token_budget:
            return None

        n_blocks = kv_len // self.compress_ratio
        pad = left_padding
        if pad is None:
            pooled = raw_k[:, : n_blocks * self.compress_ratio].reshape(
                B, n_blocks, self.compress_ratio, self.head_dim
            )
        else:
            # A row's blocks start at its first real token (Blaizzy/mlx-vlm#2028).
            cols = (
                pad[:, None, None]
                + (mx.arange(n_blocks) * self.compress_ratio)[None, :, None]
                + mx.arange(self.compress_ratio)[None, None, :]
            ).reshape(B, -1)
            # A block that runs past the buffer is masked out below.
            cols = mx.minimum(cols, kv_len - 1)
            pooled = mx.take_along_axis(
                raw_k,
                mx.broadcast_to(cols[..., None], (*cols.shape, self.head_dim)),
                axis=1,
            ).reshape(B, n_blocks, self.compress_ratio, self.head_dim)
        pooled = self.k_layernorm(
            pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype)
        )

        pooled = self.block_rope(pooled)

        q_pos = _positions(offset, S)
        q = self.q_layernorm(q).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=offset)

        # scores: sum over heads of relu(q.k), per block
        scores = mx.einsum(
            "bhsd,bnd->bsnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0).sum(axis=-1) / math.sqrt(self.head_dim)

        # Only blocks fully in the query's past; `q_pos` excludes the padding.
        n_complete = mx.maximum(q_pos + 1, 0) // self.compress_ratio
        visible = mx.arange(n_blocks)[None, None, :] < n_complete[..., None]
        scores = mx.where(visible, scores, -mx.inf)

        k = min(self.block_topk, n_blocks)
        top = mx.argpartition(-scores, k - 1, axis=-1)[..., :k]  # (B, S, k)
        picked = mx.take_along_axis(visible, top, axis=-1)

        # Blocks to tokens; a trailing partial block is never picked.
        if pad is None:
            keep_block = mx.put_along_axis(
                mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_),
                mx.where(picked, top, n_blocks),
                mx.array(True),
                axis=-1,
            )[..., :n_blocks]
            keep = mx.repeat(keep_block, self.compress_ratio, axis=-1)
            rest = kv_len - n_blocks * self.compress_ratio
            if rest:
                keep = mx.concatenate(
                    [keep, mx.zeros((B, S, rest), dtype=mx.bool_)], axis=-1
                )
        else:
            # Each row has its own block grid: scatter the picks on the token axis.
            tok = (
                pad[:, None, None, None]
                + top[..., None] * self.compress_ratio
                + mx.arange(self.compress_ratio)[None, None, None, :]
            ).reshape(B, S, -1)
            flag = mx.broadcast_to(
                picked[..., None], (*picked.shape, self.compress_ratio)
            ).reshape(B, S, -1) & (tok < kv_len)
            keep = mx.put_along_axis(
                mx.zeros((B, S, kv_len + 1), dtype=mx.bool_),
                mx.where(flag, tok, kv_len),
                flag,
                axis=-1,
            )[..., :kv_len]

        # Keep each query's own partial block: an all-masked row would attend to
        # every key, future ones included.
        own_start = n_complete * self.compress_ratio
        if pad is not None:
            own_start = own_start + pad[:, None]
        tokens = mx.arange(kv_len)
        # Physical column of each query in the shared buffer.
        kv_pos = mx.arange(kv_len - S, kv_len)
        own = (tokens[None, None, :] >= own_start[..., None]) & (
            tokens[None, None, :] <= kv_pos[None, :, None]
        )
        return (keep | own)[:, None]  # (B, 1, S, kv_len)


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        d = args.hidden_size
        # q_proj also carries the output gate: n_heads * head_dim * 2
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(
            int(args.head_dim * args.partial_rotary_factor), base=args.rope_theta
        )
        self.indexer = QSAIndexer(args)

    def __call__(self, x, mask, cache) -> mx.array:
        B, S, _ = x.shape
        # `cache` holds the attention keys and values, then the indexer keys.
        kv_cache, idx_cache = (
            (cache[0], cache[1]) if cache is not None else (None, None)
        )
        offset = kv_cache.offset if kv_cache is not None else 0

        sparse = self.indexer(x, idx_cache, offset, _left_padding(kv_cache))

        q, gate = mx.split(self.q_proj(x).reshape(B, S, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, S, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)

        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)

        if sparse is not None:
            # Combine as booleans; "causal" needs an explicit mask first.
            if mask is None:
                mask = sparse
            elif isinstance(mask, str):  # "causal"
                kv_len = k.shape[2]
                rinds = mx.arange(kv_len)
                linds = mx.arange(kv_len - S, kv_len)[:, None]
                mask = (linds >= rinds) & sparse
            elif mask.dtype == mx.bool_:
                mask = mask & sparse
            else:
                neg = mx.finfo(mask.dtype).min
                mask = mask + mx.where(sparse, mx.array(0, mask.dtype), neg)

        out = scaled_dot_product_attention(
            q, k, v, cache=kv_cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out * mx.sigmoid(gate))


class GatedDeltaNet(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_v = args.linear_num_value_heads
        self.n_k = args.linear_num_key_heads
        self.dk = args.linear_key_head_dim
        self.dv = args.linear_value_head_dim
        self.key_dim = self.dk * self.n_k
        self.value_dim = self.dv * self.n_v
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        d = args.hidden_size

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )
        # unlike qwen3-next, the projections are split
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
        self.dt_bias = mx.ones(self.n_v)
        self.A_log = mx.zeros(self.n_v)
        self.norm = RMSNormGated(
            self.dv, eps=args.rms_norm_eps, activation=args.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def __call__(self, x, mask, cache) -> mx.array:
        B, S, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.n_v, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        conv_state = (
            cache[0]
            if (cache is not None and cache[0] is not None)
            else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            cache[0] = _keep_last(conv_input, self.conv_kernel_size - 1, cache.lengths)
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1)
        q = q.reshape(B, S, self.n_k, self.dk)
        k = k.reshape(B, S, self.n_k, self.dk)
        v = v.reshape(B, S, self.n_v, self.dv)

        inv_scale = self.dk**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.top_k - 1, axis=-1)[..., : self.top_k]
        w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        out = (self.switch_mlp(x, idx) * w[..., None]).sum(axis=-2).astype(x.dtype)
        return out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class GatedResidual(nn.Module):
    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.d = args.hidden_size
        self.lowrank = args.hc_lowrank
        self.use_combine = use_combine
        hc_dim = self.hc * self.d
        self.hc_norm = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        # The last hc rows are the checkpoint's block inject: one matmul for both.
        down = self.lowrank + (self.hc if use_combine else 0)
        self.input_mix_weight_down = nn.Linear(hc_dim, down, bias=False)
        self.input_mix_weight_up = nn.Linear(self.lowrank, hc_dim, bias=False)

    def __call__(self, hyper: mx.array):
        normed = self.hc_norm(hyper)
        down = self.input_mix_weight_down(normed)
        w = nn.silu(down[..., : self.lowrank] / self.hc)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        w = w.reshape(*w.shape[:-1], self.hc, self.d)
        mixed = (w * normed.reshape(*normed.shape[:-1], self.hc, self.d)).mean(axis=-2)
        if not self.use_combine:
            return mixed
        inject = 2 * mx.sigmoid(down[..., self.lowrank :] / self.hc)
        return mixed, hyper, inject


_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(v: int) -> int:
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NGramEmbedding(nn.Module):
    """N-gram hash tables, one per head (~51B parameters)."""

    def __init__(self, args: TextArgs, embed_dim: int, ple_layer_index: int = 0):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = embed_dim // self.ngram_heads

        sizes, offsets, total = [], [], 0
        for h in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + h
            s = _nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
            sizes.append(s)
            offsets.append(total)
            total += s
        self.head_vocab_sizes = sizes
        # One table per head keeps each checkpoint file small (~6 GB in bf16).
        self.ngram_embedding = [nn.Embedding(s, head_dim) for s in sizes]

        mults = []
        max_long = (1 << 63) - 1
        half = max(1, (max_long // max(args.vocab_size, 1)) // 2)
        base_seed = args.seed + _PRIME_1 * ple_layer_index
        for i in range(self.ngram_size):
            mults.append(
                2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % half) + 1
            )
        # Not parameters: rebuilt from the config, checked in `sanitize`.
        self._mults = mx.array(mults, dtype=mx.int64)
        self._sizes = mx.array(sizes, dtype=mx.int64)
        self._offsets = mx.array(offsets, dtype=mx.int64)

    def _ngram_tokens(self, history: mx.array) -> list:
        """x[t], x[t-1], ... for the new tokens; `history` starts with the n-1
        tokens before them. Older tokens past an EOS read as EOS."""
        n, eos = self.ngram_size, self.eos_token_id
        new = history.shape[1] - (n - 1)
        # HF's `_shift_right_ignore_eos`, at the new positions only.
        tokens = [history[:, n - 1 :], history[:, n - 2 : n - 2 + new]]
        for s in range(2, n):
            older = history[:, n - 1 - s : n - 1 - s + new]
            tokens.append(mx.where(tokens[-1] == eos, eos, older))
        return tokens

    def __call__(self, ids: mx.array, cache) -> mx.array:
        n = self.ngram_size
        prev = cache[3] if cache is not None else None
        if prev is None:
            prev = mx.full((ids.shape[0], n - 1), self.eos_token_id, ids.dtype)
        # Right padded prompts work; left padded ones do not.
        history = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[3] = _keep_last(history, n - 1, cache.lengths)
        tokens = self._ngram_tokens(history.astype(mx.int64))

        # Step s gives the hash of the (s+1)-gram.
        mixed, rows = tokens[0] * self._mults[0], []
        for s in range(1, n):
            mixed = mx.bitwise_xor(mixed, tokens[s] * self._mults[s])
            lo = (s - 1) * self.heads_per_ngram
            rows.append(mixed[..., None] % self._sizes[lo : lo + self.heads_per_ngram])
        rows = mx.concatenate(rows, axis=-1)
        return mx.concatenate(
            [table(rows[..., h]) for h, table in enumerate(self.ngram_embedding)],
            axis=-1,
        )


_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)\.")


class PLELayer(nn.Module):
    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.d = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.d * self.hc
        self.ple_embedding = NGramEmbedding(args, args.ple_embed_dim, ple_layer_index)
        k = args.ple_conv_kernel_size
        self.dilation = args.ngram_size
        self.short_conv_state_len = (k - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.d, bias=False)
        self.norm_key = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_query = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_dim, group_size=self.d, eps=args.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_dim,
            hc_dim,
            kernel_size=k,
            groups=hc_dim,
            dilation=self.dilation,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        S = x.shape[1]
        n = self.short_conv_state_len
        state = (
            cache[2]
            if (cache is not None and cache[2] is not None)
            else mx.zeros((x.shape[0], n, x.shape[-1]), dtype=x.dtype)
        )
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = _keep_last(full, n, cache.lengths)
        return nn.silu(self.conv1d(full[:, -(n + S) :, :]))

    def __call__(self, hidden: mx.array, ids: mx.array, cache) -> mx.array:
        emb = self.ple_embedding(ids, cache).astype(hidden.dtype)
        key = self.norm_key(self.key_proj(emb))
        key = key.reshape(*key.shape[:-1], self.hc, self.d)
        value = self.value_proj(emb)
        query = self.norm_query(hidden)
        query = query.reshape(*query.shape[:-1], self.hc, self.d)

        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.d)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*gated.shape[:-2], -1)
        return gated + self._short_conv(self.norm_conv(gated), cache)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        ple_idx = (
            args.ple_layer_ids.index(layer_idx + 1)
            if (layer_idx + 1) in args.ple_layer_ids
            else None
        )
        self.ple = PLELayer(args, ple_idx) if ple_idx is not None else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(self, h, mask, conv_mask, cache, ids):
        if self.ple is not None:
            h = h + self.ple(h, ids, cache)

        x, hyper, inject = self.attn_hyper_connection(h)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
        else:
            x = self.self_attn(x, mask, cache)
        h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)

        x, hyper, inject = self.mlp_hyper_connection(h)
        x = self.mlp(x)
        return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # no final `norm` in this model: this mixer carries it
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ple_layers = [
            i for i in range(args.num_hidden_layers) if (i + 1) in args.ple_layer_ids
        ]

    def __call__(self, ids: mx.array, cache=None, input_embeddings=None):
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        full_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "full_attention"
        ]
        lin_idx = [
            i for i, l in enumerate(self.layers) if l.layer_type == "linear_attention"
        ]
        attn_cache = cache[full_idx[0]] if full_idx else None
        mask = create_attention_mask(
            h, attn_cache[0] if attn_cache is not None else None
        )
        # The deltanet reads in order, so it needs the left padding zeroed out.
        conv_mask = create_ssm_mask(h, cache[lin_idx[0]]) if lin_idx else None

        h = mx.tile(h, (1, 1, self.hc))
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, conv_mask, c, ids)
        return self.hyper_connection_mixer(h)


def _merge_linears(weights: dict, parts: list, target: str):
    """Concatenate the layers `parts` into `target` along the output axis,
    with their scales and biases when quantized."""
    for name in ("weight", "scales", "biases"):
        keys = [f"{p}.{name}" for p in parts]
        if all(k in weights for k in keys):
            weights[f"{target}.{name}"] = mx.concatenate(
                [weights.pop(k) for k in keys], axis=-2
            )


def _split_rows(shards: list, sizes: list) -> list:
    """Split a table stored as equal row shards into tables of `sizes` rows."""
    rows = shards[0].shape[0]
    tables, start = [], 0
    for size in sizes:
        end = start + size
        first, last = start // rows, (end - 1) // rows
        pieces = [
            s[max(start - i * rows, 0) : min(end - i * rows, rows)]
            for i, s in enumerate(shards[first : last + 1], first)
        ]
        tables.append(mx.concatenate(pieces))
        start = end
    return tables


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.text.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for i, t in enumerate(self.args.text.layer_types):
            if t == "full_attention":
                caches.append(CacheList(KVCache(), KVCache()))
            else:
                # The deltanet conv and recurrent state; a PLE layer adds its
                # conv state and n-gram context.
                num_states = 4 if i in self.model.ple_layers else 2
                caches.append(ArraysCache(num_states))
        return caches

    # Norms stored as `w` for a `1 + w` scale; `linear_attn.norm` is not one.
    _FOLD_ONE = (
        "q_layernorm.weight",
        "k_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "hc_norm.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
    )

    def sanitize(self, weights):
        # Fold the norms' `1 +` into the weight once: converted checkpoints have it.
        fold = any(k.startswith("model.language_model.") for k in weights)

        out = {}
        for k, v in weights.items():
            # Not in this text-only port: the MTP head and the vision tower.
            if k.startswith(("mtp.", "model.mtp.")):
                continue
            if k.startswith(("model.visual.", "visual.", "vision_tower.")):
                continue

            # The original checkpoint nests the text model under
            # `model.language_model.`, mlx-vlm under `language_model.`.
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            elif k.startswith("language_model."):
                k = k[len("language_model.") :]

            # The n-gram hash constants: check them against the config, then drop.
            if k.endswith(self._NGRAM_CONSTANTS):
                self._check_ngram_constant(k, v)
                continue

            # Rename the n-gram shards to `shards.{i}`, as mlx-vlm saves them.
            k = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1.", k)

            # The checkpoint stacks the experts as (E, out, in); gate_up_proj holds
            # gate, then up, on axis -2.
            if k.endswith("mlp.experts.gate_up_proj"):
                base = k[: -len("experts.gate_up_proj")]
                mid = v.shape[-2] // 2
                out[base + "switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                out[base + "switch_mlp.up_proj.weight"] = v[..., mid:, :]
                continue
            if k.endswith("mlp.experts.down_proj"):
                base = k[: -len("experts.down_proj")]
                out[base + "switch_mlp.down_proj.weight"] = v
                continue

            # torch (C, 1, K) to mlx (C, K, 1); a converted weight has shape[1] > 1.
            if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
                v = v.transpose(0, 2, 1)

            if fold and k.endswith(self._FOLD_ONE):
                v = 1.0 + v

            out[k] = v

        # Merge the checkpoint's block inject into the mix down.
        for k in [k for k in out if k.endswith("block_inject_weight.weight")]:
            p = k[: -len("block_inject_weight.weight")]
            down = p + "input_mix_weight_down"
            _merge_linears(out, [down, p + "block_inject_weight"], down)
        # The checkpoint has equal row shards; the model keeps one table per head.
        for k in [k for k in out if k.endswith("ngram_embedding.shards.0.weight")]:
            p = k[: -len("shards.0.weight")]
            n = sum(q.startswith(p + "shards.") and q.endswith(".weight") for q in out)
            shards = [out.pop(f"{p}shards.{i}.weight") for i in range(n)]
            layer = int(k.split(".layers.")[1].split(".")[0])
            sizes = self.model.layers[layer].ple.ple_embedding.head_vocab_sizes
            for h, table in enumerate(_split_rows(shards, sizes)):
                out[f"{p}{h}.weight"] = table
        return out

    _NGRAM_CONSTANTS = (
        "ple_embedding.layer_multipliers",
        "ple_embedding.ngram_heads_vocab_sizes",
        "ple_embedding.ngram_heads_offsets",
    )

    def _check_ngram_constant(self, key: str, value: mx.array) -> None:
        layer = int(key.split(".layers.")[1].split(".")[0])
        table = self.model.layers[layer].ple.ple_embedding
        rebuilt = {
            "layer_multipliers": table._mults,
            "ngram_heads_vocab_sizes": table._sizes,
            "ngram_heads_offsets": table._offsets,
        }[key.rsplit(".", 1)[1]]
        if not mx.array_equal(rebuilt, value.astype(rebuilt.dtype)):
            raise ValueError(
                f"{key}: the n-gram hash constants rebuilt from the config do "
                f"not match the checkpoint ({rebuilt.tolist()} vs "
                f"{value.tolist()}); check `seed` and the hash parameters"
            )

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # The MoE router stays in full precision.
            return not path.endswith("mlp.gate")

        return predicate
