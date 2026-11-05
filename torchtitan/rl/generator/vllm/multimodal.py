# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image prompts in the vLLM generator: the TorchTitan vision encoder behind vLLM's
multimodal interface.

Requests carry a `Prompt`: token ids with each image's placeholder run, and the
`PromptImage`s (content key, patch grid, placeholder range and the image file's bytes).
TitanRL preprocesses the images, not vLLM: ``VLLMEngine`` decodes and resizes each image
file with its `ImageSpec` (``VLLMEngine.Config.image_spec``) and submits vLLM's multimodal
input directly (`image_prompt`), so vLLM's Hugging Face processor never runs.

An image's embedding depends on the vision weights, which train, and vLLM's encoder cache
keeps it after its requests finish (only ``reset_encoder_cache`` clears it, not a prefix
cache reset). So the image's mm hash is its content key salted like the prefix cache with
the request's pinned policy version (``key@version``): vLLM's encoder and prefix caches
reuse an image's embedding only among requests pinned to the same version, e.g. the samples
of a GRPO group, and never across versions. With ``reset_kv_cache_on_model_state_dict_load``
there is no salt; the engine resets both caches on every weight load instead:

    PromptImage(key, ..., data)                     rollout worker -> generator: the file's bytes
                                                    (Geometry3K median 14 KB)
      -> decode_image_files: uint8 (C, H, W)        each TP rank decodes + resizes the file
                                                    (Geometry3K: 2.2 ms median, 12.5 ms max)
      -> vLLM mm_input(prompt ids, {"images", "image_grid_thw"}, hashes=key[@version],
                       placeholders)
      -> embed_multimodal: patchify on GPU + vision encoder, once per mm hash while cached
      -> embed_input_ids: token embeddings with image embeddings at the placeholders
      -> forward(inputs_embeds, 3D M-RoPE positions)

Normalizing and patchifying on the GPU instead of the CPU keeps ~20 ms per image off
the engine loop thread and sends 8x fewer bytes host to device; the values are the
ones `patchify` computes on the CPU.

The processor registered with vLLM only serves vLLM's startup: the spec's largest
image (`largest_image_grid`) sizes the encoder budget, and a dummy image of that size
profiles the encoder's memory.

Supported models: Qwen3.5-style (``Qwen35Model``): a ``vision_encoder`` behind
``get_vision_embeds``, merged patches as placeholder tokens, and ``MRoPE.Config``
attention. `validate_image_model` checks this before the engine is built.
"""

from __future__ import annotations

import functools
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import torch
from torch.distributed.fsdp import FSDPModule
from vllm.config import VllmConfig
from vllm.inputs import mm_input, MultiModalInput
from vllm.model_executor.models.interfaces import (
    SupportsEncoderCudaGraph,
    SupportsMRoPE,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    ProcessorInputs,
    TimingContext,
)
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from torchtitan.components.data.multimodal.mrope import get_mrope_positions

from torchtitan.components.data.multimodal.utils.image import (
    ImageSpec,
    largest_image_grid,
    patchify,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.qwen3_5.rope import MRoPE
from torchtitan.rl.generator.vllm.model_wrapper import VLLMModelWrapper
from torchtitan.rl.rollout.types import Prompt, PromptImage

# The keyword arguments vLLM batches per image and passes to `embed_multimodal`.
_IMAGE_FIELDS = {
    # uint8 (C, H, W); images of different sizes stay a list.
    "images": MultiModalFieldConfig.batched("image"),
    # (3,) raw patch grid, stacked to (num_images, 3).
    "image_grid_thw": MultiModalFieldConfig.batched("image"),
}


def _image_grids(mm_kwargs: dict[str, Any]) -> list[list[int]]:
    """The batch's raw patch grids ``[t, h, w]`` as host ints."""
    grids = mm_kwargs["image_grid_thw"]
    return grids if isinstance(grids, list) else grids.tolist()


def _pad_segment_ids(buffer: torch.Tensor, segment_ids: torch.Tensor) -> None:
    """Fill an encoder graph's ``segment_ids`` buffer: padding patches belong to no image (-1)."""
    buffer.fill_(-1)
    buffer[: segment_ids.shape[0]].copy_(segment_ids)


def validate_image_model(model_config: Decoder.Config, spec: ImageSpec) -> None:
    """Raise unless ``model_config`` can serve images preprocessed with ``spec``.

    Runs before the engine builds the model: image prompts need a vision encoder
    whose patching matches ``spec``, and M-RoPE attention.
    """
    encoder = getattr(model_config, "vision_encoder", None)
    if encoder is None:
        raise ValueError(
            f"{type(model_config).__qualname__} has no vision encoder; build the "
            "generator model with one, or unset VLLMEngine.Config.image_spec."
        )
    if (
        encoder.patch_size,
        encoder.temporal_patch_size,
        encoder.spatial_merge_size,
    ) != (
        spec.patch_size,
        spec.temporal_patch_size,
        spec.merge_size,
    ):
        raise ValueError(
            f"The image spec has patch_size={spec.patch_size}, "
            f"temporal_patch_size={spec.temporal_patch_size}, merge_size={spec.merge_size}; "
            f"the vision encoder uses {encoder.patch_size}, {encoder.temporal_patch_size}, "
            f"{encoder.spatial_merge_size}. Use the model's preprocessing in "
            "VLLMEngine.Config.image_spec."
        )
    mrope_hf_config(model_config)


def mrope_hf_config(model_config: Decoder.Config) -> dict[str, Any]:
    """The HF config entries that make vLLM compute 3D M-RoPE positions.

    With them vLLM calls the model's ``get_mrope_input_positions``; text-only
    generators keep 1D positions.
    """
    attention = model_config.first_full_attention
    rope = getattr(attention, "rope", None)
    if not isinstance(rope, MRoPE.Config):
        raise ValueError(
            "VLLMEngine.Config.image_spec needs a model whose attention uses "
            f"MRoPE.Config (Qwen3.5); got {type(rope).__qualname__}."
        )
    # Presence required: an mrope_section switches vLLM to 3D positions.
    return {
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": rope.theta,
            "mrope_section": list(rope.mrope_section),
        }
    }


def image_prompt(
    prompt: Prompt,
    pixels: Sequence[torch.Tensor],
    *,
    cache_salt: str | None = None,
) -> MultiModalInput:
    """Build vLLM's multimodal engine input for a prompt whose images are loaded.

    Args:
        prompt: Token ids with each image's placeholder run, and the images.
        pixels: Each image's resized uint8 image ``(C, H, W)``, in ``prompt.images`` order.
        cache_salt: vLLM prefix-cache salt: the request's pinned policy version, the
            oldest one its KV may come from. It also salts the image hashes
            (``key@salt``), because an image's encoder output depends on the vision
            weights, which train. None (the engine resets its caches on every weight
            load): the hashes are the bare content keys.
    """
    kwargs = MultiModalKwargsItems.from_hf_inputs(
        {
            "images": list(pixels),
            "image_grid_thw": torch.tensor([image.grid_thw for image in prompt.images]),
        },
        _IMAGE_FIELDS,
    )
    return mm_input(
        prompt_token_ids=prompt.token_ids,
        mm_kwargs=kwargs,
        mm_hashes={
            "image": [
                image.key if cache_salt is None else f"{image.key}@{cache_salt}"
                for image in prompt.images
            ]
        },
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=image.offset, length=image.length)
                for image in prompt.images
            ]
        },
        cache_salt=cache_salt,
    )


class _ImageProcessingInfo(BaseProcessingInfo):
    """Sizes vLLM's encoder budget and memory profiling: the spec's largest
    image that fits ``max_model_len``."""

    def __init__(self, ctx: InputProcessingContext, *, image_spec: ImageSpec) -> None:
        super().__init__(ctx)
        self.image_spec = image_spec

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def largest_grid(self, seq_len: int) -> tuple[int, int, int]:
        return largest_image_grid(self.image_spec, seq_len)

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        t, h, w = self.largest_grid(seq_len)
        return {"image": t * h * w // self.image_spec.merge_size**2}


class _ImageDummyInputsBuilder(BaseDummyInputsBuilder[_ImageProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        _, h, w = self.info.largest_grid(seq_len)
        patch_size = self.info.image_spec.patch_size
        return {
            "image": self._get_dummy_images(
                width=w * patch_size,
                height=h * patch_size,
                num_images=mm_counts.get("image", 0),
            )
        }


class _ImageProcessor(BaseMultiModalProcessor[_ImageProcessingInfo]):
    """Bypasses vLLM's image preprocessing: TitanRL preprocesses images itself.

    vLLM normally runs the model's Hugging Face processor on every prompt (decode, resize,
    normalize, patchify, expand placeholders). Here ``VLLMEngine`` decodes and resizes each
    image file with the renderer's `ImageSpec`, `image_prompt` submits the result to vLLM
    directly, and ``embed_multimodal`` patchifies on the GPU. So this processor builds only
    the dummy inputs vLLM profiles with at startup.
    """

    # Prompts arrive as token ids and images preprocessed, so no tokenizer is needed
    # (the engine runs with skip_tokenizer_init).
    requires_tokenizer: ClassVar[bool] = False

    def _get_mm_fields_config(
        self, hf_inputs: Any, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _IMAGE_FIELDS

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        """One zero image of the spec's largest size per dummy image item."""
        num_images = inputs.mm_data_items.get_count("image", strict=False)
        grid = self.info.largest_grid(self.info.ctx.model_config.max_model_len)
        spec = self.info.image_spec
        length = grid[0] * grid[1] * grid[2] // spec.merge_size**2
        image = torch.zeros(
            3, grid[1] * spec.patch_size, grid[2] * spec.patch_size, dtype=torch.uint8
        )
        images = tuple(
            PromptImage(
                key=f"profile-{index}",
                path=f"profile-{index}",
                grid_thw=grid,
                offset=index * length,
                length=length,
                data=b"",  # unused: the pixels are `image`
            )
            for index in range(num_images)
        )
        prompt = Prompt(token_ids=[0] * (num_images * length), images=images)
        return image_prompt(prompt, [image] * num_images)


class VLLMMultimodalModelWrapper(
    VLLMModelWrapper, SupportsMultiModal, SupportsMRoPE, SupportsEncoderCudaGraph
):
    """`VLLMModelWrapper` that keeps the vision encoder and serves image prompts.

    Implements vLLM's multimodal (`embed_multimodal`, `embed_input_ids`) and M-RoPE
    (`get_mrope_input_positions`) interfaces for Qwen3.5-style models: a
    ``vision_encoder`` reached through the model's ``get_vision_embeds``, merged
    patches as placeholder tokens, and 3D M-RoPE positions. vLLM feeds multimodal
    models embeddings, also for text-only steps, and 3D positions ``(3, num_tokens)``.

    Also implements vLLM's encoder CUDA graph interface, used when
    ``VLLMCudaGraphConfig.vision_encoder`` is set: vLLM captures
    `encoder_cudagraph_forward` once per token budget and replays it for each step's new
    images, packed into the smallest budget that fits and padded to it.
    """

    supports_multimodal: ClassVar[Literal[True]] = True
    supports_encoder_cudagraph: ClassVar[Literal[True]] = True

    def __init__(self, *, image_spec: ImageSpec, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._image_spec = image_spec
        # FSDP makes the first module that runs forward its root. This wrapper calls
        # the embeddings, the vision encoder and the layers directly, never the root
        # model, so make the root the whole model before any of them runs. Otherwise
        # the grouped [tok_embeddings, norm, lm_head] state raises ("requires a single
        # root module") when embed_input_ids runs first.
        # TODO: use a public FSDP lazy-init API once one exists.
        self.model._get_fsdp_state()._lazy_init()

    def embed_multimodal(
        self, *, images: torch.Tensor | list[torch.Tensor], image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """vLLM required API. Encode a batch of images, one embedding block per image.

        Args:
            images: Resized uint8 images ``(C, H, W)`` on the GPU, stacked when they
                share a size.
            image_grid_thw: ``(num_images, 3)`` raw patch grids.
        """
        pixel_values = patchify(list(images), self._image_spec)
        with self.parallelism_context.activate_spmd():
            vision_embeds, num_tokens = self.model.get_vision_embeds(
                pixel_values, grid_thw=image_grid_thw
            )
        return vision_embeds.split(num_tokens.tolist())

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        """vLLM required API for encoder CUDA graphs: the buffers each graph reads."""
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixel_values", "learned_pos", "rope_cache", "segment_ids"],
            out_hidden_size=self.model.vision_encoder.merger.linear_fc2.weight.shape[0],
            padding_logics={"segment_ids": _pad_segment_ids},
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        """vLLM required API for encoder CUDA graphs: the smallest and largest token
        budgets to capture. vLLM captures powers of 2 in between."""
        spec = self._image_spec
        # The smallest image the spec resizes to, in merged tokens.
        min_tokens = max(1, spec.min_pixels // (spec.patch_size * spec.merge_size) ** 2)
        max_tokens = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return min_tokens, max(min_tokens, max_tokens)

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        """vLLM required API for encoder CUDA graphs: each image's patches and tokens."""
        merge_unit = self._image_spec.merge_size**2
        return [
            EncoderItemSpec(input_size=t * h * w, output_tokens=t * h * w // merge_unit)
            for t, h, w in _image_grids(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        """vLLM required API for encoder CUDA graphs: the images at ``indices``."""
        images, grids = mm_kwargs["images"], _image_grids(mm_kwargs)
        return {
            "images": [images[i] for i in indices],
            "image_grid_thw": [grids[i] for i in indices],
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        """vLLM required API for encoder CUDA graphs: a budget's buffers, holding one
        zero image of exactly ``token_budget`` tokens."""
        encoder = self.model.vision_encoder
        merge = self._image_spec.merge_size
        num_patches = token_budget * merge**2
        pixel_values = torch.zeros(
            num_patches, encoder.patch_embed.in_features, device=device
        )
        return EncoderCudaGraphCaptureInputs(
            values=self._encoder_graph_inputs(
                pixel_values, [[1, merge, token_budget * merge]]
            )
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        """vLLM required API for encoder CUDA graphs: the batch's unpadded inputs, which
        vLLM copies into the graph's buffers (zero-padded; ``segment_ids`` with -1)."""
        pixel_values = patchify(list(mm_kwargs["images"]), self._image_spec)
        return EncoderCudaGraphReplayBuffers(
            values=self._encoder_graph_inputs(pixel_values, _image_grids(mm_kwargs))
        )

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        """vLLM required API for encoder CUDA graphs: the forward vLLM captures."""
        encoder = self._unsharded_vision_encoder()
        with self.parallelism_context.activate_spmd():
            return encoder.forward_padded(
                inputs["pixel_values"],
                learned_pos=inputs["learned_pos"],
                rope_cache=inputs["rope_cache"],
                segment_ids=inputs["segment_ids"],
            )

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        """vLLM required API for encoder CUDA graphs: images larger than every budget."""
        images = mm_kwargs["images"]
        grid_thw = torch.tensor(_image_grids(mm_kwargs), device=images[0].device)
        return torch.cat(self.embed_multimodal(images=images, image_grid_thw=grid_thw))

    def _encoder_graph_inputs(
        self, pixel_values: torch.Tensor, grids: list[list[int]]
    ) -> dict[str, torch.Tensor]:
        """`Qwen35VisionEncoder.forward_padded`'s inputs for ``grids``, unpadded."""
        encoder = self._unsharded_vision_encoder()
        with self.parallelism_context.activate_spmd():
            learned_pos, rope_cache = encoder.compute_position_embeddings(grids)
        segment_ids = torch.repeat_interleave(
            torch.arange(len(grids), dtype=torch.int32),
            torch.tensor([t * h * w for t, h, w in grids]),
        )
        return {
            "pixel_values": pixel_values.to(encoder.patch_embed.weight.dtype),
            "learned_pos": learned_pos,
            "rope_cache": rope_cache,
            "segment_ids": segment_ids.to(pixel_values.device),
        }

    def _unsharded_vision_encoder(self) -> Any:
        """The vision encoder with its parameters unsharded.

        FSDP unshards a module around its ``forward`` only, and these methods use other
        entries. The generator never reshards, and weight loads unshard into the same
        buffers, so this is a no-op once vLLM's startup profiling has run the encoder.
        """
        encoder = self.model.vision_encoder
        assert isinstance(encoder, FSDPModule)
        encoder.unshard()
        return encoder

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Sequence[torch.Tensor] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """vLLM required API. Token embeddings with image embeddings written over the
        placeholder tokens that ``is_multimodal`` marks."""
        inputs_embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """vLLM required API. The decoder on precomputed embeddings.

        Args:
            input_ids: Unused; vLLM passes embeddings to multimodal models.
            positions: ``(3, num_tokens)`` M-RoPE positions.
            inputs_embeds: ``(num_tokens, dim)`` from `embed_input_ids`.
        """
        assert inputs_embeds is not None and positions is not None
        # TorchTitan's MRoPE takes (num_tokens, 3).
        positions = positions.transpose(0, 1)
        with self.parallelism_context.activate_spmd():
            # Mirrors the model's forward after embedding (layers, then the final norm).
            # TODO: share this loop with Decoder.forward / Qwen35Model.forward.
            h = inputs_embeds
            layer_kwargs = self._prepare_step_kwargs(h.shape[0])
            for layer in self.model.layers.values():
                h = layer(h, None, positions, **layer_kwargs)
            h = self.model.norm(h)
        return self._replicated_local(h)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """vLLM required API: vLLM calls it once per request of an M-RoPE model for the
        prompt's ``(3, num_tokens)`` positions and the offset ``position - token index`` of
        every token after the prompt.

        `get_mrope_positions` computes them from each image's placeholder offset and grid,
        as the batcher does for the trainer, so the generator's and the trainer's positions
        match.
        """
        vision_items = []
        for feature in mm_features:
            # `image_prompt` submits only images, and the processor cache is off, so
            # vLLM keeps each feature's data.
            assert (
                feature.modality == "image" and feature.data is not None
            ), feature.modality
            t, h, w = feature.data["image_grid_thw"].data.tolist()
            vision_items.append((feature.mm_position.offset, (t, h, w)))
        positions = get_mrope_positions(
            len(input_tokens),
            vision_items,
            spatial_merge_size=self._image_spec.merge_size,
        )
        delta = int(positions.max()) + 1 - len(input_tokens)
        return positions.transpose(0, 1), delta


def register_image_processor(model_cls: type, image_spec: ImageSpec) -> None:
    """Register TorchTitan's image processor (`_ImageProcessor`) with vLLM for
    ``model_cls``, instead of the Hugging Face processor vLLM would run.

    vLLM requires a registered processor for a multimodal model, but uses this one only
    at startup: ``image_spec``'s largest image sizes the encoder budget and profiles
    memory. Prompts bypass it (`image_prompt`).
    """
    MULTIMODAL_REGISTRY.register_processor(
        _ImageProcessor,
        info=functools.partial(_ImageProcessingInfo, image_spec=image_spec),
        dummy_inputs=_ImageDummyInputsBuilder,
    )(model_cls)
