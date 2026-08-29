"""Bidirectional multi-head co-attention (paper eq. 5-7).

The authors' `FMNVD.py` imports this module but it was never uploaded to their
repo, so it is reconstructed here from the equations:

    Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    F_t' = LayerNorm(F_t + Attn(a -> t))    # title attends to audio
    F_a' = LayerNorm(F_a + Attn(t -> a))    # audio attends to title

The constructor / call signature is fixed by the authors' call site:

    co_attention(d_k, d_v, n_heads, dropout, d_model,
                 visual_len, sen_len, fea_v, fea_s, pos=False)
    out_v, out_s = module(v=..., s=..., v_len=..., s_len=...)

`v` is the visual stream in the original paper; in our title+audio scope it
carries the **audio/speech** stream, and `s` carries the **title** stream.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, only used when pos=True."""

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadCrossAttention(nn.Module):
    """One direction of the co-attention: query stream attends to context stream."""

    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_heads * d_k)
        self.w_k = nn.Linear(d_model, n_heads * d_k)
        self.w_v = nn.Linear(d_model, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, context_mask=None):
        """query: (B, Lq, d_model), context: (B, Lc, d_model).

        context_mask: (B, Lc) bool/float, 1 = keep. Returns (B, Lq, d_model).
        """
        B, Lq, _ = query.shape
        Lc = context.size(1)

        q = self.w_q(query).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(context).view(B, Lc, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(context).view(B, Lc, self.n_heads, self.d_v).transpose(1, 2)

        # (B, H, Lq, Lc)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if context_mask is not None:
            mask = context_mask.to(dtype=torch.bool)[:, None, None, :]
            # A row that is fully masked (e.g. an empty transcript) would give
            # softmax(-inf) = NaN, so keep those rows unmasked and zero the
            # attended output afterwards instead.
            all_masked = ~mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~mask & ~all_masked, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, Lq, d_v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.n_heads * self.d_v)
        out = self.dropout(self.fc(out))

        if context_mask is not None:
            # Nothing to attend to -> contribute nothing (pure residual).
            keep = context_mask.to(dtype=out.dtype).sum(dim=1).clamp(max=1.0)
            out = out * keep[:, None, None]
        return out


class co_attention(nn.Module):
    """Bidirectional co-attention between the `v` and `s` streams.

    Naming (lower-case class, positional args) follows the authors' call site.
    """

    def __init__(self, d_k, d_v, n_heads, dropout, d_model,
                 visual_len, sen_len, fea_v, fea_s, pos=False):
        super().__init__()
        self.d_model = d_model
        self.visual_len = visual_len
        self.sen_len = sen_len
        self.pos = pos

        # Input feature dims may differ from d_model; identity when they match.
        self.proj_v = nn.Linear(fea_v, d_model) if fea_v != d_model else nn.Identity()
        self.proj_s = nn.Linear(fea_s, d_model) if fea_s != d_model else nn.Identity()

        if pos:
            self.pos_v = PositionalEncoding(d_model, visual_len)
            self.pos_s = PositionalEncoding(d_model, sen_len)

        # s -> v : v queries s   (audio attends to title)
        self.attn_s2v = MultiHeadCrossAttention(d_model, d_k, d_v, n_heads, dropout)
        # v -> s : s queries v   (title attends to audio)
        self.attn_v2s = MultiHeadCrossAttention(d_model, d_k, d_v, n_heads, dropout)

        self.norm_v = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)

    def forward(self, v, s, v_len=None, s_len=None):
        """v: (B, Lv, fea_v), s: (B, Ls, fea_s).

        v_len / s_len may be an int sequence length, a (B,) tensor of lengths, or
        a (B, L) attention mask. Returns (v_out, s_out), both (B, L*, d_model).
        """
        v = self.proj_v(v)
        s = self.proj_s(s)

        if self.pos:
            v = self.pos_v(v)
            s = self.pos_s(s)

        v_mask = self._as_mask(v_len, v)
        s_mask = self._as_mask(s_len, s)

        v_out = self.norm_v(v + self.attn_s2v(v, s, s_mask))  # eq. 7
        s_out = self.norm_s(s + self.attn_v2s(s, v, v_mask))  # eq. 6
        return v_out, s_out

    @staticmethod
    def _as_mask(length, x):
        """Normalise v_len/s_len into a (B, L) keep-mask, or None."""
        if length is None:
            return None
        B, L = x.size(0), x.size(1)
        if not torch.is_tensor(length):  # plain int: every position valid
            return None if int(length) >= L else torch.cat(
                [x.new_ones(B, int(length)), x.new_zeros(B, L - int(length))], dim=1
            )
        if length.dim() == 2:  # already an attention mask
            return length
        # (B,) lengths -> mask
        ar = torch.arange(L, device=x.device)[None, :]
        return (ar < length[:, None]).to(x.dtype)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from utils import get_device

    torch.manual_seed(42)
    device = get_device()
    B, TITLE_LEN, SPEECH_LEN, DIM, HEADS = 4, 32, 64, 128, 4

    module = co_attention(
        d_k=DIM // HEADS, d_v=DIM // HEADS, n_heads=HEADS, dropout=0.1,
        d_model=DIM, visual_len=SPEECH_LEN, sen_len=TITLE_LEN,
        fea_v=DIM, fea_s=DIM, pos=False,
    ).to(device).eval()

    v = torch.randn(B, SPEECH_LEN, DIM, device=device)   # audio/speech stream
    s = torch.randn(B, TITLE_LEN, DIM, device=device)    # title stream
    v_mask = torch.ones(B, SPEECH_LEN, device=device)
    v_mask[3] = 0                                        # sample 3: empty transcript
    s_mask = torch.ones(B, TITLE_LEN, device=device)

    with torch.no_grad():
        v_out, s_out = module(v=v, s=s, v_len=v_mask, s_len=s_mask)

    print(f"device                : {device}")
    print(f"heads                 : {HEADS}   pos={module.pos}   dropout=0.1")
    print(f"params                : {sum(p.numel() for p in module.parameters()):,}")
    print()
    print(f"input  v (audio)      : {tuple(v.shape)}")
    print(f"input  s (title)      : {tuple(s.shape)}")
    print(f"output v_out          : {tuple(v_out.shape)}")
    print(f"output s_out          : {tuple(s_out.shape)}")
    assert v_out.shape == v.shape, "v_out shape mismatch"
    assert s_out.shape == s.shape, "s_out shape mismatch"
    print("SHAPES MATCH: v_out == v, s_out == s")
    print(f"finite outputs        : {bool(torch.isfinite(v_out).all() and torch.isfinite(s_out).all())}")
    print(f"empty-transcript row 3 residual-only (s_out[3] == LayerNorm(s[3])): "
          f"{torch.allclose(s_out[3], module.norm_s(s[3]), atol=1e-5)}")
