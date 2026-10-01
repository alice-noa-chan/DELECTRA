"""Optimized causal attention that reuses ELECTRA's existing Q/K/V weights."""

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def causal_padding_mask(attention_mask: Tensor) -> Tensor | None:
    """Return a visible-key mask, or None for fully packed causal sequences.

    SDPA interprets True as allowed. Combine right-padding with the causal
    triangle for padded inputs. Unpadded blocks use ``is_causal=True`` without
    allocating a [batch, heads, sequence, sequence] attention score matrix.
    """
    if attention_mask.bool().all():
        return None
    length = attention_mask.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=attention_mask.device)
    return causal.tril()[None, None] & attention_mask[:, None, None, :].bool()


class ElectraSDPAAttention(nn.Module):
    """Replace attention computation without replacing ELECTRA parameters.

    Existing Q/K/V modules are reused with identical state-dict names. Output
    projection, residuals, LayerNorm and feed-forward blocks remain untouched.
    ``sdpa`` lets PyTorch choose a kernel; ``flash`` permits only its CUDA
    FlashAttention kernel and raises instead of silently using math attention.
    """

    def __init__(self, original: nn.Module, backend: str) -> None:
        super().__init__()
        if backend not in {"sdpa", "flash"}:
            raise ValueError("optimized attention backend must be sdpa or flash")
        if original.position_embedding_type != "absolute":
            raise ValueError("optimized attention supports ELECTRA absolute positions")
        self.query = original.query
        self.key = original.key
        self.value = original.value
        self.dropout = original.dropout
        self.num_attention_heads = original.num_attention_heads
        self.attention_head_size = original.attention_head_size
        self.all_head_size = original.all_head_size
        self.backend = backend

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        if (
            head_mask is not None
            or encoder_hidden_states is not None
            or past_key_values is not None
            or output_attentions
        ):
            raise ValueError(
                "optimized causal attention excludes head masks, cross-attention, "
                "cache and attention weights"
            )
        batch, length, _ = hidden_states.shape

        def split_heads(projection: nn.Module) -> Tensor:
            # [B, L, hidden] -> [B, heads, L, head_width]. No weights are changed.
            return (
                projection(hidden_states)
                .view(batch, length, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )

        query, key, value = (
            split_heads(layer) for layer in (self.query, self.key, self.value)
        )
        if self.backend == "flash":
            if query.device.type != "cuda" or query.dtype not in {
                torch.float16,
                torch.bfloat16,
            }:
                raise ValueError("flash attention requires CUDA FP16/BF16 Q/K/V")
            if attention_mask is not None:
                raise ValueError(
                    "flash attention requires unpadded blocks in this adapter"
                )
        context = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            if self.backend == "flash"
            else nullcontext()
        )
        with context:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                # SDPA applies dropout even in eval unless explicitly set to zero.
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        return attended.transpose(1, 2).contiguous().view(
            batch, length, self.all_head_size
        ), None
