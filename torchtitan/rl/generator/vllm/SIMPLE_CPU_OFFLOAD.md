# Local CPU KV offload

Use vLLM's [SimpleCPUOffloadConnector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)
to retain KV in each worker's CPU DRAM and reload it after GPU cache eviction.
It uses the same `VLLMEngine.Config.kv_transfer_config` API as
[Mooncake Store](MOONCAKE_STORE.md), without a Store service or RDMA setup.

Before launching RL, override Titan's allocator defaults. The tested vLLM version
rejects expandable allocations with any KV connector, including SimpleCPU:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

```python
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="SimpleCPUOffloadConnector",
    kv_role="kv_both",
    kv_connector_extra_config={"cpu_bytes_to_use_per_rank": 1 << 30},
)
```

`cpu_bytes_to_use_per_rank` reserves CPU cache capacity **per GPU worker**;
TP=2 allocates 2 GiB in this example. Leave room for trainer weight buffers and
other host processes. vLLM owns allocation, content-hash lookup, LRU eviction,
and asynchronous copies. Its default eager mode saves eligible blocks during
generation and at request completion, including single-turn requests.

Keep prefix caching enabled. Use `generator_router.hot_swap=False` and
`generator.reset_kv_cache_on_model_state_dict_load=False`. Titan salts KV with
the loaded policy version, rejects requests before the first trainer weight
load, and drains remaining transfers after user outputs finish.

Each TP scheduler queries its local CPU cache. The shared
[KV offload scheduler](cpu_kv_offload.py) applies transfer completions and block
release in the same order on every TP rank:

```text
Local CPU prefix lookup -> allocate GPU blocks -> each worker loads its KV shard
                                             -> collect all TP completions
                                             -> resume requests / release blocks
```

DP replicas and independent generators have separate CPU caches. Use sticky
routing for reuse between turns; moving a session can require prefill again.
`session_kv_holder` can retain GPU KV independently. Hybrid prefix reuse follows the upstream connector's
available cache boundaries; unmatched tokens are recomputed.

### Session release

Titan adds the same session lifecycle to both CPU offload connectors:

```text
Rollout/session or group ends
  -> router notifies every generator (including previous routing destinations)
  -> engine waits for completed requests and their outstanding save batch
  -> connector.evict_blocks(released_hashes, active_hashes)
```

SimpleCPU moves unused CPU blocks to the front of the eviction queue; their KV
remains reusable until reclaimed. Shared prefixes and block aliases used by
other sessions on this DP replica keep their priority. GPU session holding is
not required. Unrelated later saves do not delay the release.

This is a one-shot hint for cached/pending KV, not a tombstone. Future requests
or lazy offload may cache those blocks again. [Mooncake](MOONCAKE_STORE.md)
uses the same notification to attempt deletion from its shared Store instead.

An eight-GPU example is
`torchtitan_recipes.tests.rl.alphabet_sort.rl_grpo_qwen3_5_debug_varlen_cpu_offload`.
Standalone `generate.py` is not supported yet. Disk offload and CUDA graphs
have not been validated for this integration.
