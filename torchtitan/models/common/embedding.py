# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder_sharding import dense_activation_placement


class Embedding(nn.Embedding, Module):
    """Configurable embedding with optional local vocab-parallel execution.

    With a TP group, each rank looks up only its vocab shard and returns a
    TP-``Partial`` output; the caller owns the reduction.
    ``VocabParallelEmbedding`` performs it explicitly.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int
        padding_idx: int | None = None

    def __init__(self, config: Config):
        super().__init__(
            config.num_embeddings, config.embedding_dim, padding_idx=config.padding_idx
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run vocab-parallel embedding when the active mesh has a TP group."""
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            return F.embedding(
                input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

        tp_size = dist.get_world_size(tp_group)
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
        offset = dist.get_rank(tp_group) * chunk_size
        mask = (input >= offset) & (input < offset + self.weight.shape[0])
        local_input = (input - offset).clamp(0, self.weight.shape[0] - 1)
        # padding_idx is global; only its owning shard should suppress gradients.
        local_padding_idx = None
        if (
            self.padding_idx is not None
            and offset <= self.padding_idx < offset + self.weight.shape[0]
        ):
            local_padding_idx = self.padding_idx - offset
        out = F.embedding(
            local_input,
            self.weight,
            local_padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out * mask.unsqueeze(-1).to(out.dtype)


class VocabParallelEmbedding(Embedding):
    """Vocab-parallel token embedding that reduces its partial output.

    The masked vocab-shard lookup runs in a local-SPMD region that returns a
    TP-``Partial`` output. The reduction is the remat region ``<fqn>.tp_reduce``:
    ``Partial -> Shard(0)`` is a reduce-scatter under sequence parallelism, and
    ``Partial -> Invariant`` is an all-reduce without it. ``replicate_output``
    selects ``Partial -> Replicate`` instead.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Embedding.Config):
        replicate_output: bool = False
        """All-reduce the output to TP-``Replicate`` instead of the residual
        layout. Multimodal models need it: the vision scatter writes features at
        arbitrary sequence positions, so every TP rank needs the full sequence.
        """

    def __init__(self, config: Config):
        super().__init__(config)
        self.replicate_output = config.replicate_output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # The masked lookup indexes the vocab shard with bare tensor arithmetic,
        # so it runs as a local region whose output is TP-partial.
        partial = dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
        output = spmd.local_map(out_types=(partial.local_type, partial.partition_spec))(
            super().forward
        )(input)
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is None:
            return output
        if self.replicate_output:
            dst = spmd.R
        elif spmd_dense_sp_enabled():
            dst = spmd.S(0)
        else:
            dst = spmd.I
        return remat.region(
            spmd.redistribute,
            self.remat_region_name("tp_reduce"),
            recompute=self.remat_should_recompute("tp_reduce"),
        )(
            output,
            tp_group,
            src=spmd.P,
            dst=dst,
            backward_options={"op_dtype": output.dtype},
        )


__all__ = ["Embedding", "VocabParallelEmbedding"]
