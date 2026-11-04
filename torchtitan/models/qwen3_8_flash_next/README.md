# Qwen3.8-Flash-Next

Flash-Next reuses Qwen3.5 Gated DeltaNet, vision, FSDP and optimizer lifecycle.
Its model package implements four-stream gated residuals, Qwen Sparse Attention
(QSA), lexical embeddings (PLE), and the checkpoint's router rounding.

## Reference and checkpoint

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  checkpoint revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Transformers `Qwen4ExpForCausalLM`, `Qwen4ExpVisionModel`, and
  `Qwen4ExpForConditionalGeneration` at revision
  `494470d677d05d44153d235d795e0d01040307ab`.
- The adapter maps 1,627 released decoder/vision keys, including 128 numeric
  lexical shards. Pinned name/shape digests verify the complete meta inventory.
  The 31 MTP keys are excluded from offline conversion; DCP requests only mapped
  model keys. MTP is outside this training model.

## Architecture

| Path | Released shape or configuration |
| --- | --- |
| Decoder | 48 layers, model width 2,560, vocabulary 248,320 |
| Gated residual | Four streams, grouped offset RMSNorm, gate rank 320 |
| Attention | Three GDN layers followed by one QSA layer |
| QSA | 24 query / 2 KV heads, head width 256, partial MRoPE |
| Indexer | Four heads of width 128, four-token blocks, 2,048-token budget plus causal tail |
| MoE | 512 experts, top 10, routed/shared width 640, sigmoid shared gate |
| PLE | Layer 2, 16 bigram/trigram heads of width 160, gated lexical injection |
| Lexical table | 320,001,536 rows by 160 columns, 51.2B parameters |
| Vision | 27 layers, width 1,152, 16-pixel patches, merger output width 2,560 |

```text
original IDs ----> PLE hashes/lookup ----> lexical residual
                        |
embeddings [T,D] -> repeat [T,4*D] -> gated read [T,D]
                                         |
                                    GDN/QSA or MoE
                                         |
                                gated write [T,4*D]
                                         |
                            final stream mixer -> [T,D] -> LM head
```

PLE uses original token IDs and one-dimensional document positions, independently
of multimodal rotary coordinates. The hash history resets after EOS, as in
the pinned reference; EOS does not reset PLE convolution or QSA visibility.
Position zero additionally resets all three paths for packed documents, matching
separate reference executions of each document. Debug flavors use the test
tokenizer's EOS ID 2003; the released flavor uses 248044.

QSA caches complete-block/tail metadata once per input. Equal ReLU scores select
earlier blocks first, so masked future tokens cannot change past selections.
HF has an unspecified cutoff tie policy; the HF fixture requires selection
margins and separately checks natural block/expert choices.

`indexer_loss=None` freezes the indexer before FSDP and runs selection without an
autograd graph. This supports a checkpoint's fixed selector. Training from
scratch requires an explicit KL objective, as in the recipes below. Its detached
teacher and backbone inputs train only the indexer. Empty/short documents keep
finite zero indexer gradients so all DP ranks participate in FSDP reductions.
Zero gradients allow AdamW state/decay updates on entirely empty steps.

## Build and train

Supported parallelism is FSDP, TP with or without sequence parallelism, EP and
real pipeline parallelism. CP and fake pipeline process groups are rejected. TP degree must divide the two QSA KV
heads, so degrees 1 and 2 are supported.
Flavors are `debugmodel`, `debugmodel_mm`, and `125B-A6B`.

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss

config = build_model_config(
    "debugmodel",
    parallelism=ParallelismConfig(data_parallel_shard_degree=2),
    seq_len=64,
    indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
)
```

```bash
NGPU=2 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel ./run_train.sh

python -m tests.integration_tests.run_tests outputs/flash_next_mm \
  --test_suite models --execution_mode real_pg --ngpu 2 \
  --test_name qwen3_8_flash_next_mm_fsdp
```

The multimodal integration entry consumes `qwen38_flash_next_debugmodel_mm` with
image fusion, PLE and QSA at 1,024 tokens. The text fixture uses 64 tokens and ten
training steps. Reduced fixtures retain both attention types and routed/shared
experts.

TP keeps residual/PLE state replicated, shards QSA heads and lexical feature
columns, and reduces detached head probabilities before pooling each block. The
replicated indexer contributes its KL gradient once across the TP ranks.

```bash
NGPU=2 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel_tp2 ./run_train.sh
```

The eight-GPU recipe combines FSDP2, TP2, EP4 and PP2 with four virtual stages,
Interleaved1F1B and full activation remat. PLE belongs to virtual stage 1, after
token embedding. Stages exchange `[T, 4 * D]` and receive original token IDs and
document positions independently of multimodal rotary coordinates. The final
stage owns the stream mixer and LM head. The shared engine rejects SPMD
typechecking under PP; the TP/EP recipe checks types separately.

```bash
NGPU=8 MODULE=torchtitan_recipes.tests.models.qwen3_8_flash_next \
  CONFIG=qwen38_flash_next_debugmodel_all_parallelism ./run_train.sh
```

## Numerical coverage

The CPU tests cover residual/norm equations, hash and convolution boundaries,
selected-attention backward, teacher KL, frozen/empty indexer gradients, router
rounding, padding exclusion and auxiliary lifecycle under full/selective remat.
The two-GPU FSDP test composes differing document lengths, padding, accumulation
and remat. The adapter roundtrip uses 16 numbered shards to detect lexical-sort
mistakes, and verifies all released names and shapes from pinned headers.

```bash
pytest tests/unit_tests/cpu/models/qwen3_8_flash_next -q
pytest tests/unit_tests/gpu/models/qwen3_8_flash_next -q
python -m scripts.checkpoint_conversion.numerical_tests_qwen3_8_flash_next \
  --device cuda --output outputs/flash_next_hf.json
```

The HF harness runs production CUDA GDN and grouped experts without substituting
model operations. Its fixed fixtures compare text, standalone vision, and full
image/video fusion in train/eval, including logits, masked CE, available parameter
gradients, embeddings and pixels. BF16 gradient bounds are `atol=1e-4`,
`rtol=0.03125`; standalone FP32 vision uses `atol=3e-6`, `rtol=3e-5`.
Independent CPU feature equations provide the CPU coverage.

The registered real-PG GB300 suite runs the TP2/EP2/SP fixture for ten steps
and compares full-precision loss and gradient norm against its committed golden.
[Golden provenance](../../../tests/assets/losses/real_pg/gb300/qwen3_8_flash_next.provenance.json)
records the hardware, runtime, source and data hashes of the independent captures.

The fixture uses the test tokenizer EOS ID 2003. PLE also uses this ID to
fill missing n-gram history, so correcting the former 2047 default changes
lookups at document starts and requires a new numerical golden. Two fresh
independent GB300 launches and the registered export agree bitwise on all
twenty loss/gradient-norm values; the comparison remains exact.

Pipeline tests compare original-ID payloads and all stage parameter gradients
with an unsplit model.

```bash
python -m tests.integration_tests.run_tests outputs/flash_next_numerics \
  --test_suite gb300 --execution_mode real_pg --gpu_arch gb300 --ngpu 2 \
  --test_name qwen3_8_flash_next_tp+ep_numerics
```

## QSA memory and training limits

Both dense indexer scoring and gathered SDPA use nonreentrant PyTorch checkpoints
per query chunk. They preserve the original math and nest inside block
`torch_remat` policies. Visibility masks are rebuilt inside score checkpoints;
gathered K/V and MATH SDPA intermediates are recomputed during backward. Teacher
construction and auxiliary logging stay outside the checkpointed functions.
Ranking still performs quadratic work; chunking removes the quadratic saved
ReLU graph and per-token saved K/V gathers, rather than eliminating that compute.

Standalone released-geometry QSA completes ten BF16 AdamW steps on one GB300:

| Tokens | Peak allocated memory |
| --- | --- |
| 512 | 1.1 GiB |
| 1,024 | 1.9 GiB |
| 2,048 | 3.4 GiB |
| 4,096 | 3.8 GiB |

At 512 tokens, the checkpointed path preserves ten-step loss, gradient norm, KL
and final weights bitwise relative to the uncheckpointed path, whose peak is
13.0 GiB. This is a standalone QSA capacity check, not a full released-model
training claim. Full-model fixtures are the 64-token text and 1,024-token image
recipes; full released-model training, including the 262,144-token context, is
unqualified.

Attention Gym's [distinct K/V work](https://github.com/meta-pytorch/attention-gym/pull/678)
and [grouped gathered-attention kernel](https://github.com/meta-pytorch/attention-gym/pull/679)
track the replacement for the MATH path. The latter lacks the required grouped
training backward. Its adoption must preserve causal blocks/tails and Qwen's
teacher KL.

- Dynamic selections require `training.disable_accelerator_graphs=True`.
- The lexical table alone is about 95.4 GiB in BF16. FSDP shards persistent state
  and gathers its local TP feature shard for computation. At TP degree 2, each
  rank materializes about 47.7 GiB of table weights before the lookup.
- The meta inventory has 177.4B decoder/vision parameters including the lexical
  table. Full-checkpoint materialization and distributed conversion of that
  table are unqualified.
- Inference KV caching and MTP are outside this training port.
