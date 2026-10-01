"""Optional fused vocabulary loss without changing ELECTRA's model weights."""

from functools import lru_cache

import torch
from torch import Tensor, nn


@lru_cache(maxsize=1)
def _liger_loss():
    # Keep ordinary CPU use independent of Triton and its Linux/CUDA runtime.
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except ImportError as error:
        raise RuntimeError(
            "liger loss requires the optional kernels dependency"
        ) from error
    return LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)


def fused_causal_lm_loss(
    head: nn.Linear, features: Tensor, input_ids: Tensor, attention_mask: Tensor
) -> Tensor:
    """Predict x[t+1] from projected clean features[t], preserving tied weights.

    Liger consumes [valid next-token pairs, embedding width] and the original
    vocabulary weight/bias. It does not return full [B,L,V] logits. Masking must
    retain both visible endpoints, exactly as the reference PyTorch loss does.
    FP32 gradient accumulation matters when the vocabulary weights are also the
    shared input embedding table. No model monkey-patches or new weights occur.
    """
    if features.device.type != "cuda":
        raise ValueError("liger fused loss requires CUDA")
    valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    if not valid.any():
        raise ValueError("CLM requires at least one valid next-token pair")
    return _liger_loss()(
        head.weight,
        features[:, :-1][valid].contiguous(),
        input_ids[:, 1:][valid].contiguous(),
        bias=head.bias,
    )
