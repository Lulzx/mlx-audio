# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Vector quantization for Qwen3 TTS Tokenizer."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .layers import Conv1d


class EuclideanCodebook(nn.Module):
    """Euclidean codebook for vector quantization."""

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self._epsilon = 1e-5
        self._dim = dim
        self._codebook_size = codebook_size

        # EMA tracking tensors (used during training, but we load them from weights)
        self.initialized = mx.zeros([1], dtype=mx.float32)
        self.embed_sum = mx.zeros([codebook_size, dim], dtype=mx.float32)
        self.cluster_usage = mx.zeros([codebook_size], dtype=mx.float32)

        # Computed embedding (will be set in update_in_place)
        self._embedding = None
        self._c2 = None

    def update_in_place(self):
        """Update the embedding from EMA statistics."""
        cluster_usage = mx.maximum(self.cluster_usage, self._epsilon)[:, None]
        self._embedding = self.embed_sum / cluster_usage
        self._c2 = (self._embedding * self._embedding).sum(axis=-1) / 2

    def encode(self, xs: mx.array) -> mx.array:
        """Encode continuous vectors to discrete codes.

        Args:
            xs: Input tensor of shape (batch, time, dim)

        Returns:
            Codes of shape (batch, time)
        """
        if self._embedding is None:
            self.update_in_place()

        target_shape = xs.shape[:-1]
        xs = xs.reshape(-1, xs.shape[-1])  # (batch * time, dim)

        # Find nearest codebook entry using dot product
        # argmin ||x - e||^2 = argmin ||x||^2 - 2*x.e + ||e||^2
        # Since ||x||^2 is constant for each x, we minimize -x.e + ||e||^2/2
        dot_prod = xs @ self._embedding.T  # (batch * time, codebook_size)
        codes = (self._c2 - dot_prod).argmin(axis=-1)  # (batch * time,)

        return codes.reshape(target_shape)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode discrete codes to continuous vectors.

        Args:
            codes: Code tensor of shape (batch, time)

        Returns:
            Decoded vectors of shape (batch, time, dim)
        """
        if self._embedding is None:
            self.update_in_place()

        target_shape = list(codes.shape) + [self._dim]
        flat_codes = codes.flatten()
        quantized = mx.take(self._embedding, flat_codes, axis=0)
        return quantized.reshape(target_shape)


class VectorQuantizer(nn.Module):
    """Single vector quantizer with optional projection."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_dim = codebook_dim

        self.codebook = EuclideanCodebook(dim=codebook_dim, codebook_size=codebook_size)

    def encode(self, xs: mx.array) -> mx.array:
        """Encode to discrete codes.

        Args:
            xs: Input tensor of shape (batch, dim, time)

        Returns:
            Codes of shape (batch, time)
        """
        # (batch, dim, time) -> (batch, time, dim)
        xs = xs.swapaxes(-1, -2)
        return self.codebook.encode(xs)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode from discrete codes.

        Args:
            codes: Codes of shape (batch, time)

        Returns:
            Decoded tensor of shape (batch, dim, time)
        """
        xs = self.codebook.decode(codes)
        return xs.swapaxes(-1, -2)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with multiple codebooks."""

    def __init__(
        self,
        dim: int,
        input_dim: int,
        output_dim: int,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_quantizers = num_quantizers

        # Input projection
        if input_dim != codebook_dim:
            self.input_proj = Conv1d(input_dim, codebook_dim, 1, bias=False)
        else:
            self.input_proj = None

        # Output projection
        if output_dim != codebook_dim:
            self.output_proj = Conv1d(codebook_dim, output_dim, 1, bias=False)
        else:
            self.output_proj = None

        # Quantizer layers
        self.layers = [
            VectorQuantizer(
                dim=codebook_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            for _ in range(num_quantizers)
        ]

    def encode(self, xs: mx.array) -> mx.array:
        """Encode to discrete codes using residual quantization.

        Args:
            xs: Input tensor of shape (batch, input_dim, time)

        Returns:
            Codes of shape (batch, num_quantizers, time)
        """
        if self.input_proj is not None:
            xs = self.input_proj(xs)

        codes = []
        residual = xs
        for layer in self.layers:
            layer_codes = layer.encode(residual)
            quantized = layer.decode(layer_codes)
            residual = residual - quantized
            codes.append(layer_codes)

        return mx.stack(codes, axis=1)  # (batch, num_quantizers, time)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode from discrete codes.

        Args:
            codes: Codes of shape (batch, num_quantizers, time)

        Returns:
            Decoded tensor of shape (batch, output_dim, time)
        """
        quantized = None
        for i, layer in enumerate(self.layers):
            layer_quantized = layer.decode(codes[:, i])
            if quantized is None:
                quantized = layer_quantized
            else:
                quantized = quantized + layer_quantized

        if self.output_proj is not None:
            quantized = self.output_proj(quantized)

        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    """Split RVQ with semantic and acoustic quantizers (Qwen3-TTS style)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        codebook_dim: int,
        codebook_size: int,
        num_semantic_quantizers: int = 1,
        num_acoustic_quantizers: int = 31,
        valid_num_quantizers: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_semantic_quantizers = num_semantic_quantizers
        self.num_acoustic_quantizers = num_acoustic_quantizers
        self.valid_num_quantizers = valid_num_quantizers

        # Semantic RVQ (usually 1 codebook)
        self.semantic_residual_vector_quantizer = ResidualVectorQuantizer(
            dim=codebook_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            num_quantizers=num_semantic_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        # Acoustic RVQ (multiple codebooks)
        self.acoustic_residual_vector_quantizer = ResidualVectorQuantizer(
            dim=codebook_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            num_quantizers=num_acoustic_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

    def encode(self, xs: mx.array, num_quantizers: int = None) -> mx.array:
        """Encode using both semantic and acoustic quantizers.

        Args:
            xs: Input tensor of shape (batch, input_dim, time)
            num_quantizers: Number of quantizers to use (default: valid_num_quantizers)

        Returns:
            Codes of shape (batch, num_quantizers, time)
        """
        if num_quantizers is None:
            num_quantizers = self.valid_num_quantizers

        # Semantic codes
        semantic_codes = self.semantic_residual_vector_quantizer.encode(xs)

        # Acoustic codes
        acoustic_codes = self.acoustic_residual_vector_quantizer.encode(xs)

        # Concatenate and take only valid quantizers
        all_codes = mx.concatenate([semantic_codes, acoustic_codes], axis=1)

        return all_codes[:, :num_quantizers]

    def decode(self, codes: mx.array) -> mx.array:
        """Decode from codes.

        Args:
            codes: Codes of shape (batch, num_quantizers, time)

        Returns:
            Decoded tensor of shape (batch, output_dim, time)
        """
        num_codes = codes.shape[1]

        # Split into semantic and acoustic
        semantic_codes = codes[:, : self.num_semantic_quantizers]
        acoustic_codes = codes[:, self.num_semantic_quantizers :]

        # Decode semantic
        semantic_quantized = self.semantic_residual_vector_quantizer.decode(semantic_codes)

        # Decode acoustic if present
        if acoustic_codes.shape[1] > 0:
            # Pad acoustic codes if needed (decoder expects full acoustic codebooks)
            num_acoustic = acoustic_codes.shape[1]
            if num_acoustic < self.num_acoustic_quantizers:
                # Only decode the codes we have
                acoustic_quantized = None
                for i in range(num_acoustic):
                    layer_quantized = self.acoustic_residual_vector_quantizer.layers[i].decode(
                        acoustic_codes[:, i]
                    )
                    if acoustic_quantized is None:
                        acoustic_quantized = layer_quantized
                    else:
                        acoustic_quantized = acoustic_quantized + layer_quantized

                if self.acoustic_residual_vector_quantizer.output_proj is not None:
                    acoustic_quantized = self.acoustic_residual_vector_quantizer.output_proj(
                        acoustic_quantized
                    )
            else:
                acoustic_quantized = self.acoustic_residual_vector_quantizer.decode(acoustic_codes)

            return semantic_quantized + acoustic_quantized

        return semantic_quantized
