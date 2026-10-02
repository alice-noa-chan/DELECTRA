"""Decoder-only adapters around the external Transformers ELECTRA backbone."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import ElectraConfig, ElectraForPreTraining

from deletcra.attention import ElectraSDPAAttention, causal_padding_mask
from deletcra.config import ModelConfig


def electra_config(settings: ModelConfig) -> ElectraConfig:
    """Keep ELECTRA's blocks and heads, but allow attention only to the prefix.

    Transformers uses ``is_decoder`` to build the causal self-attention mask.
    Cross-attention stays disabled because there is no separate encoder here.
    Learned absolute positions, GELU, and LayerNorm remain ELECTRA defaults.
    """
    return ElectraConfig(
        vocab_size=settings.vocab_size,
        embedding_size=settings.embedding_size,
        hidden_size=settings.hidden_size,
        num_hidden_layers=settings.num_layers,
        num_attention_heads=settings.num_heads,
        intermediate_size=settings.intermediate_size,
        max_position_embeddings=settings.max_positions,
        hidden_dropout_prob=settings.dropout,
        attention_probs_dropout_prob=settings.dropout,
        pad_token_id=settings.pad_token_id,
        bos_token_id=settings.bos_token_id,
        is_decoder=True,
        add_cross_attention=False,
        use_cache=False,
        attn_implementation="eager",
        deletcra_attention_backend=settings.attention_backend,
    )


@dataclass
class DecoderOutput:
    hidden_states: Tensor
    rtd_logits: Tensor | None
    lm_logits: Tensor | None


def validate_batch(input_ids: Tensor, attention_mask: Tensor) -> None:
    """Require nonempty, right-padded sequences with a visible first token."""
    if input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("input_ids must have nonempty [batch, sequence] shape")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if attention_mask.device != input_ids.device:
        raise ValueError("input_ids and attention_mask must use the same device")
    if input_ids.device.type == "xla":
        # The production loader validates values on CPU before transferring.
        # Python conditions on device values would split every XLA graph here.
        return
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError("attention_mask must contain only zero or one")
    if not attention_mask[:, 0].bool().all():
        raise ValueError("every sequence must have a visible first token")
    if (attention_mask[:, 1:] > attention_mask[:, :-1]).any():
        raise ValueError("only right padding is supported")


class CausalElectra(nn.Module):
    """Causal ELECTRA with an RTD head and a separate, tied vocabulary head.

    RTD logits at t judge the token at t. LM logits at t predict the token at
    t+1. Cross-attention is absent; both heads use the same causal backbone.
    """

    def __init__(self, settings: ModelConfig | ElectraConfig) -> None:
        super().__init__()
        config = (
            electra_config(settings)
            if isinstance(settings, ModelConfig)
            else deepcopy(settings)
        )
        config.is_decoder = True
        config.add_cross_attention = False
        config.use_cache = False
        config._attn_implementation = "eager"
        if config.bos_token_id is None or config.bos_token_id == config.pad_token_id:
            raise ValueError("set a BOS token distinct from PAD")
        self.config = config
        self.attention_backend = getattr(config, "deletcra_attention_backend", "eager")
        if self.attention_backend not in {"eager", "sdpa", "flash"}:
            raise ValueError("unsupported causal attention backend")
        pretrained = ElectraForPreTraining(config)
        self.electra = pretrained.electra
        self.rtd_head = pretrained.discriminator_predictions
        # The discriminator head scores the current token as original/replaced.
        # A separate vocabulary head predicts the next token. Project to the
        # embedding width first so its output weights can reuse the token table.
        self.lm_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.embedding_size),
            nn.GELU(),
            nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps),
        )
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        for layer in (self.lm_projection[0], self.lm_head):
            nn.init.normal_(layer.weight, std=config.initializer_range)
            nn.init.zeros_(layer.bias)
        self.lm_head.weight = self.electra.embeddings.word_embeddings.weight
        if self.attention_backend != "eager":
            for layer in self.electra.encoder.layer:
                layer.attention.self = ElectraSDPAAttention(
                    layer.attention.self, self.attention_backend
                )

    def share_generator_embeddings(self, generator: "CausalElectra") -> None:
        """Tie token/position embeddings and retie the generator vocabulary head.

        Generator CLM gradients then reach discriminator embeddings directly.
        Call before constructing an optimizer; deduplicate joint parameters.
        Independently saved models preserve values but not cross-model aliases.
        """
        for name in (
            "vocab_size",
            "embedding_size",
            "max_position_embeddings",
            "pad_token_id",
            "bos_token_id",
        ):
            if getattr(self.config, name) != getattr(generator.config, name):
                raise ValueError(f"shared embeddings require matching {name}")
        main, other = self.electra.embeddings, generator.electra.embeddings
        if main.word_embeddings.weight.device != other.word_embeddings.weight.device:
            raise ValueError("shared embeddings require matching devices")
        other.word_embeddings = main.word_embeddings
        other.position_embeddings = main.position_embeddings
        # Replacing the input table does not automatically update the old output
        # alias. Retie it explicitly so generation uses the newly shared weights.
        generator.lm_head.weight = main.word_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        compute_lm: bool = True,
        compute_rtd: bool = True,
        logits_to_keep: int = 0,
    ) -> DecoderOutput:
        """Run the backbone and requested heads; zero keeps all LM positions.

        A positive logits_to_keep projects only the final positions into the
        vocabulary. Hidden states and optional RTD outputs still cover the full
        input. This reduces decoding work without changing attention or weights.
        """
        if (
            not isinstance(logits_to_keep, int)
            or isinstance(logits_to_keep, bool)
            or logits_to_keep < 0
        ):
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        validate_batch(input_ids, attention_mask)
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if self.attention_backend == "eager":
            hidden = self.electra(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
        else:
            # Reuse ELECTRA embeddings, optional width projection, and all of
            # its encoder blocks. Only mask preparation and attention change.
            # Avoid HF's eager-only additive mask on fully packed flash inputs.
            hidden = self.electra.embeddings(input_ids=input_ids)
            if hasattr(self.electra, "embeddings_project"):
                hidden = self.electra.embeddings_project(hidden)
            hidden = self.electra.encoder(
                hidden,
                attention_mask=causal_padding_mask(attention_mask),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        lm_hidden = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
        return DecoderOutput(
            hidden_states=hidden,
            rtd_logits=self.rtd_head(hidden) if compute_rtd else None,
            lm_logits=self.lm_head(self.lm_projection(lm_hidden))
            if compute_lm
            else None,
        )

    @torch.no_grad()
    def generate(
        self,
        prefix: Tensor,
        max_new_tokens: int = 8,
        *,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Greedy generation with optional EOS stopping for instruction replies.

        Return [batch, prefix + generated] IDs. Completed rows get right padding
        while other rows finish; stop when all rows produce EOS. Without an EOS
        ID, preserve the original fixed-length behavior. The LM head needs
        training before these tokens can be useful completions.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if prefix.ndim != 2 or prefix.shape[1] < 1:
            raise ValueError("prefix must have nonempty [batch, sequence] shape")
        if (prefix == self.config.pad_token_id).any():
            raise ValueError("generation requires unpadded prefixes")
        if prefix.shape[1] + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError("generation would exceed max_position_embeddings")
        if eos_token_id is not None and (
            isinstance(eos_token_id, bool)
            or not isinstance(eos_token_id, int)
            or not 0 <= eos_token_id < self.config.vocab_size
            or eos_token_id in {self.config.pad_token_id, self.config.bos_token_id}
        ):
            raise ValueError("EOS must be a vocabulary ID distinct from PAD and BOS")
        was_training = self.training
        self.eval()
        try:
            tokens = prefix.clone()
            finished = torch.zeros(len(prefix), dtype=torch.bool, device=prefix.device)
            for _ in range(max_new_tokens):
                logits = (
                    self(
                        tokens,
                        tokens.ne(self.config.pad_token_id),
                        compute_rtd=False,
                        logits_to_keep=1,
                    )
                    .lm_logits[:, -1]
                    .clone()
                )
                special_ids = [self.config.pad_token_id, self.config.bos_token_id]
                logits[:, special_ids] = -torch.inf
                next_ids = logits.argmax(-1, keepdim=True)
                next_ids[finished] = self.config.pad_token_id
                tokens = torch.cat((tokens, next_ids), dim=1)
                if eos_token_id is not None:
                    finished |= next_ids[:, 0].eq(eos_token_id)
                    if finished.all():
                        break
            return tokens
        finally:
            self.train(was_training)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)
        torch.save(self.state_dict(), directory / "model.pt")

    @classmethod
    def load(
        cls, directory: str | Path, *, attention_backend: str | None = None
    ) -> "CausalElectra":
        """Restore weights, optionally choosing a compatible execution backend.

        Use an explicit ``sdpa`` override to inspect flash-trained weights on
        CPU. This changes the computation kernel, not the saved parameters.
        """
        directory = Path(directory)
        config = ElectraConfig.from_pretrained(directory, local_files_only=True)
        if attention_backend is not None:
            config.deletcra_attention_backend = attention_backend
        model = cls(config)
        model.load_state_dict(
            torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        )
        return model

    @classmethod
    def from_encoder_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        bos_token_id: int,
        local_files_only: bool = True,
    ) -> "CausalElectra":
        """Preserve discriminator weights, change attention, initialize an LM head.

        Downloads are opt-in. A converted bidirectional checkpoint still needs
        causal training; its newly initialized LM head is not a trained LM.
        """
        config = ElectraConfig.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        config.bos_token_id = bos_token_id
        model = cls(config)
        original, info = ElectraForPreTraining.from_pretrained(
            checkpoint,
            config=model.config,
            local_files_only=local_files_only,
            output_loading_info=True,
        )
        if info["missing_keys"] or info["unexpected_keys"] or info["mismatched_keys"]:
            raise ValueError("checkpoint must contain a complete ELECTRA discriminator")
        model.electra.load_state_dict(original.electra.state_dict())
        model.rtd_head.load_state_dict(original.discriminator_predictions.state_dict())
        return model
