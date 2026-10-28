"""Compressed historical-clip memory for causal tactile prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _MemoryReadLayer(nn.Module):
    """Let current tactile queries read a fixed bank of past-only tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.attention_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(
        self,
        current: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        has_history: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_norm(current)
        normalized_memory = self.memory_norm(memory)
        update, _ = self.cross_attention(
            query=query,
            key=normalized_memory,
            value=normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        active = has_history[:, None, None].to(dtype=update.dtype)
        current = current + self.attention_scale.to(update.dtype) * update * active
        ffn_update = self.ffn(self.ffn_norm(current))
        return current + self.ffn_scale.to(ffn_update.dtype) * ffn_update * active


class CausalClipMemory(nn.Module):
    """Compress previous clips and condition only the current clip on them.

    Historical queries have shape ``(B,K,T,2,Q,D)`` and are independently
    reduced to ``M`` tokens per clip. The current clip can read those tokens,
    but no operation sends current information back into historical slots.
    This gives clip-level causality while leaving the existing within-clip
    temporal reasoner unchanged.
    """

    def __init__(
        self,
        embed_dim: int,
        history_clips: int = 2,
        memory_tokens_per_clip: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if history_clips < 1:
            raise ValueError("history_clips must be positive")
        if memory_tokens_per_clip < 1:
            raise ValueError("memory_tokens_per_clip must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.embed_dim = int(embed_dim)
        self.history_clips = int(history_clips)
        self.memory_tokens_per_clip = int(memory_tokens_per_clip)
        self.memory_queries = nn.Parameter(
            torch.randn(1, memory_tokens_per_clip, embed_dim) * 0.02
        )
        self.recency_embedding = nn.Parameter(
            torch.randn(1, history_clips, 1, embed_dim) * 0.02
        )
        self.source_norm = nn.LayerNorm(embed_dim)
        self.compressor_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(embed_dim * mlp_ratio)
        self.compressor_norm = nn.LayerNorm(embed_dim)
        self.compressor_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.read_layers = nn.ModuleList(
            [
                _MemoryReadLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    residual_scale=residual_scale,
                )
                for _ in range(num_layers)
            ]
        )

    def compress_clips(self, queries: torch.Tensor) -> torch.Tensor:
        """Reduce ``(...,T,2,Q,D)`` tactile queries to ``(...,M,D)``."""
        if queries.ndim < 5 or queries.shape[-3] != 2:
            raise ValueError(
                "Expected clip queries with shape (...,T,2,Q,D), got "
                f"{tuple(queries.shape)}"
            )
        if queries.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected query width {self.embed_dim}, got {queries.shape[-1]}"
            )
        leading_shape = queries.shape[:-4]
        num_frames = queries.shape[-4]
        position = torch.arange(
            num_frames,
            device=queries.device,
            dtype=torch.float32,
        )[:, None]
        frequency = torch.exp(
            torch.arange(
                0,
                self.embed_dim,
                2,
                device=queries.device,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / self.embed_dim)
        )
        temporal_encoding = torch.zeros(
            num_frames,
            self.embed_dim,
            device=queries.device,
            dtype=torch.float32,
        )
        temporal_encoding[:, 0::2] = torch.sin(position * frequency)
        temporal_encoding[:, 1::2] = torch.cos(
            position * frequency[: temporal_encoding[:, 1::2].shape[1]]
        )
        temporal_shape = [1] * queries.ndim
        temporal_shape[-4] = num_frames
        temporal_shape[-1] = self.embed_dim
        positioned_queries = queries + temporal_encoding.to(queries.dtype).reshape(
            temporal_shape
        )
        source = positioned_queries.reshape(
            -1,
            num_frames * 2 * queries.shape[-2],
            self.embed_dim,
        )
        seeds = self.memory_queries.expand(source.shape[0], -1, -1)
        normalized_source = self.source_norm(source)
        update, _ = self.compressor_attention(
            query=seeds,
            key=normalized_source,
            value=normalized_source,
            need_weights=False,
        )
        memory = seeds + update
        memory = memory + self.compressor_ffn(self.compressor_norm(memory))
        memory = self.output_norm(memory)
        return memory.reshape(
            *leading_shape,
            self.memory_tokens_per_clip,
            self.embed_dim,
        )

    def _validate_history(
        self,
        history_memory: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> tuple[int, int]:
        if history_memory.ndim != 4:
            raise ValueError(
                "history_memory must have shape (B,K,M,D), got "
                f"{tuple(history_memory.shape)}"
            )
        batch_size, history_clips, memory_tokens, width = history_memory.shape
        expected = (
            batch_size,
            self.history_clips,
            self.memory_tokens_per_clip,
            self.embed_dim,
        )
        if tuple(history_memory.shape) != expected:
            raise ValueError(
                f"Expected history memory shape {expected}, got "
                f"{tuple(history_memory.shape)}"
            )
        if history_valid.shape != (batch_size, history_clips):
            raise ValueError(
                f"Expected history_valid shape {(batch_size, history_clips)}, got "
                f"{tuple(history_valid.shape)}"
            )
        return batch_size, history_clips

    def read_memory(
        self,
        current_queries: torch.Tensor,
        history_memory: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Condition current queries on already-compressed historical clips."""
        if current_queries.ndim != 5 or current_queries.shape[2] != 2:
            raise ValueError(
                "Expected current queries with shape (B,T,2,Q,D), got "
                f"{tuple(current_queries.shape)}"
            )
        batch_size, _ = self._validate_history(history_memory, history_valid)
        if current_queries.shape[0] != batch_size:
            raise ValueError("Current and history batch sizes must match")

        memory = history_memory + self.recency_embedding
        memory = memory.reshape(
            batch_size,
            self.history_clips * self.memory_tokens_per_clip,
            self.embed_dim,
        )
        padding_mask = ~history_valid.to(device=memory.device, dtype=torch.bool)
        padding_mask = padding_mask[:, :, None].expand(
            -1, -1, self.memory_tokens_per_clip
        ).reshape(batch_size, -1)
        has_history = ~padding_mask.all(dim=1)

        # MultiheadAttention produces NaNs for an all-masked row. Expose one
        # harmless token for those rows, then gate the complete update to zero.
        safe_padding_mask = padding_mask.clone()
        safe_padding_mask[~has_history, 0] = False

        current_shape = current_queries.shape
        current = current_queries.reshape(batch_size, -1, self.embed_dim)
        for layer in self.read_layers:
            current = layer(
                current,
                memory,
                safe_padding_mask,
                has_history,
            )
        return current.reshape(current_shape)

    def forward(
        self,
        current_queries: torch.Tensor,
        history_valid: torch.Tensor,
        history_queries: torch.Tensor | None = None,
        history_memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return memory-conditioned queries and this clip's cacheable memory."""
        if (history_queries is None) == (history_memory is None):
            raise ValueError(
                "Provide exactly one of history_queries or history_memory"
            )
        if history_queries is not None:
            if history_queries.ndim != 6:
                raise ValueError(
                    "history_queries must have shape (B,K,T,2,Q,D), got "
                    f"{tuple(history_queries.shape)}"
                )
            history_memory = self.compress_clips(history_queries)
        assert history_memory is not None
        conditioned = self.read_memory(
            current_queries,
            history_memory,
            history_valid,
        )
        current_memory = self.compress_clips(current_queries)
        return conditioned, current_memory


__all__ = ["CausalClipMemory"]
