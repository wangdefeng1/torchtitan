# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent multi-rank generators sharing a real Mooncake CPU store.

Run explicitly with RUN_MOONCAKE_TEST=1. Requires mooncake-transfer-engine,
mooncake_master on PATH, and idle GPUs with working GPUDirect RDMA (up to eight for TP x DP/EP).
"""

import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import time
import traceback
import uuid
from unittest.mock import patch

import pytest
import torch


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _generator_worker(
    device_id,
    port,
    connector_args,
    hold_sessions,
    block_size,
    commands,
    results,
    rank,
    tp,
    dp,
    ep,
    async_scheduling,
    prefix_match_unit,
):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=device_id,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(tp * dp),
        VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="0",
    )
    try:
        import torch.distributed as dist

        from torchtitan.config import DebugConfig

        if prefix_match_unit is None:
            from torchtitan.models.qwen3 import build_model_config
        else:
            from torchtitan.models.qwen3_5 import build_model_config
        from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
        from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
        from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
        from torchtitan.rl.generator.vllm.engine import VLLMEngine
        from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
        from torchtitan.rl.generator.vllm.worker import TorchTitanGPUModelRunner
        from torchtitan.tools.utils import has_cuda_capability
        from vllm.config import KVTransferConfig
        from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group

        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
        parallelism = InferenceParallelismConfig(
            tensor_parallel_degree=tp,
            data_parallel_degree=dp,
            expert_parallel_degree=ep,
        )
        model = build_model_config(
            "debugmodel_moe" if ep > 1 else "debugmodel",
            parallelism=parallelism.to_training(),
            seq_len=2048,
        )
        model.local_compile.regions = []
        if tp > 1:
            # Exercise shared pending-lookup decisions as well as hit/miss results.
            connector_args["kv_connector_extra_config"]["lookup_async"] = True
        engine = VLLMEngine.Config(
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            kv_transfer_config=KVTransferConfig(**connector_args),
            session_kv_holder=(VLLMSessionKVHolder.Config() if hold_sessions else None),
            # Pre-SM90 engines already select 256-token blocks. Exercise that
            # granularity on newer GPUs too, without overriding an owned arg.
            extra_engine_args={
                "async_scheduling": async_scheduling,
                **(
                    {
                        "prefix_match_unit": prefix_match_unit,
                        "mamba_cache_mode": "align",
                    }
                    if prefix_match_unit is not None
                    else {}
                ),
                **(
                    {"block_size": block_size}
                    if block_size is not None and has_cuda_capability(9, 0)
                    else {}
                ),
            },
        ).build(
            model=model,
            parallelism=parallelism,
            model_path="tests/assets/tokenizer",
            max_num_seqs=4,
            model_dtype="bfloat16",
            gpu_memory_limit=0.15,
            max_num_batched_tokens=1024,
            checkpointer=None,
            dist_moe_runtime=None,
            debug=DebugConfig(seed=42),
            reset_kv_cache_on_model_state_dict_load=False,
            rank=rank,
            generator_name=f"generator-{device_id}",
            output_dir="/tmp",
        )
        # Use real model weights in the normal pinned receive buffers.
        torch.manual_seed(42)
        for tensor in engine.prefetched_model_state_dict.values():
            if tensor.ndim == 1:
                tensor.fill_(1.0)
            else:
                tensor.normal_(mean=0.0, std=0.02)
        engine.load_prefetched_model_state_dict(0)
        scheduler = engine._engine.engine_core.engine_core.scheduler
        store_scheduler = scheduler.connector.connector_scheduler
        lookup_enabled = store_scheduler.enable_lookup
        block_size = scheduler.block_size
        if prefix_match_unit is not None:
            assert scheduler.mamba_partial_cache_hit
        results.put(
            ("ready", (block_size, str(torch.cuda.get_device_properties(rank).uuid)))
        )
        while (command := commands.get()) is not None:
            version, prompt, clear_local, request_id, max_tokens, force_cold = command
            if version != engine._loaded_policy_version:
                # Change real weights, not just the version label.
                for tensor in engine.prefetched_model_state_dict.values():
                    tensor.mul_(0.75)
                engine.load_prefetched_model_state_dict(version)
            # A cold reference bypasses both cache tiers without changing weights.
            store_scheduler.enable_lookup = lookup_enabled and not force_cold
            if clear_local or force_cold:
                engine.release_sessions(["session-0"], [])
                assert engine._engine.reset_prefix_cache(reset_connector=False)
            request = EngineRequest(
                request_id=request_id,
                prompt_token_ids=prompt,
                # Leave room for the uncomputed final token and async scheduling
                # lag beyond a complete block, so reuse proves decode offloading.
                sampling=SamplingConfig(
                    temperature=0.0, max_tokens=max_tokens, seed=42
                ),
                group_id=0,
                routing_session_id="session-0",
                arrival_time=time.time(),
            )
            # Keep the group pinned to 0 even after updating weights.
            request.min_policy_version = 0
            # Idle DP replicas still participate in EP forwards and global progress.
            assert engine.add_requests([request] if prompt is not None else []) == []
            scheduler = engine._engine.engine_core.engine_core.scheduler
            if tp > 1 and prompt is not None:
                local_id = scheduler.local_request_ids[request_id]
                ids = [None] * tp
                dist.all_gather_object(ids, local_id, group=get_tp_group().cpu_group)
                assert len(set(ids)) == tp and request_id not in ids
            finished = []
            num_tail_saves = 0
            no_forward = TorchTitanGPUModelRunner.kv_connector_no_forward

            def track_no_forward(scheduler_output, vllm_config):
                nonlocal num_tail_saves
                assert scheduler_output.total_num_scheduled_tokens == 0
                num_tail_saves += sum(
                    bool(req.can_save and req.boundary_state_offloads)
                    for req in scheduler_output.kv_connector_metadata.requests
                )
                return no_forward(scheduler_output, vllm_config)

            deadline = time.monotonic() + 60
            # Keep stepping background transfers through the same engine contract
            # used by EngineLoop, including after the last request output finishes.
            with patch.object(
                TorchTitanGPUModelRunner,
                "kv_connector_no_forward",
                side_effect=track_no_forward,
            ):
                while engine.has_unfinished_requests():
                    assert time.monotonic() < deadline, "KV transfer failed to drain"
                    finished.extend(engine.step())
            if prefix_match_unit is not None:
                assert (
                    not scheduler.connector.connector_scheduler.has_pending_push_work()
                )
                assert all(
                    block.ref_cnt == 0
                    for block in scheduler.kv_cache_manager.block_pool.blocks
                    if not block.is_null
                )
            if engine._coordinate_mooncake:
                assert not scheduler.local_request_ids
                assert not scheduler.logical_request_ids
            if prompt is None:
                results.put(("idle", None))
                continue
            assert len(finished) == 1
            if hold_sessions:
                assert engine._session_kv_holder.num_sessions == 1
                held = engine._session_kv_holder._held["session-0"]
                assert held and all(block.ref_cnt > 0 for block in held)
            output = finished[0]
            results.put(
                (
                    "output",
                    (
                        output.token_ids,
                        output.token_logprobs,
                        output.metrics_inputs.num_cached_tokens,
                        num_tail_saves,
                    ),
                )
            )
        engine.shutdown()
        engine._engine.engine_core.shutdown()
        cleanup_dist_env_and_memory()
    except BaseException:
        results.put(("error", traceback.format_exc()))
        raise


@pytest.mark.skipif(
    os.getenv("RUN_MOONCAKE_TEST") != "1",
    reason="opt-in Mooncake RDMA integration test",
)
@pytest.mark.parametrize(
    "roles, save_decode_cache, hold_sessions, block_size, tp, dp, ep, async_scheduling, prefix_match_unit",
    [
        pytest.param(
            ("kv_both", "kv_both"), True, False, 256, 1, 1, 1, True, None, id="decode"
        ),
        pytest.param(
            ("kv_both", "kv_both"),
            False,
            False,
            None,
            1,
            1,
            1,
            True,
            None,
            id="prompt-only",
        ),
        pytest.param(
            ("kv_producer", "kv_consumer"),
            None,
            False,
            None,
            1,
            1,
            1,
            True,
            None,
            id="roles",
        ),
        pytest.param(
            ("kv_both", "kv_both"),
            True,
            True,
            None,
            1,
            1,
            1,
            True,
            None,
            id="session-holder",
        ),
        pytest.param(
            ("kv_both", "kv_both"), True, False, None, 2, 1, 1, True, None, id="tp2"
        ),
        pytest.param(
            ("kv_both", "kv_both"),
            True,
            False,
            None,
            2,
            1,
            1,
            False,
            None,
            id="tp2-sync",
        ),
        pytest.param(
            ("kv_both", "kv_both"), True, False, None, 1, 2, 2, True, None, id="dp2-ep2"
        ),
        pytest.param(
            ("kv_both", "kv_both"),
            True,
            False,
            None,
            2,
            2,
            4,
            True,
            None,
            id="tp2-dp2-ep4",
        ),
        pytest.param(
            ("kv_both", "kv_both"),
            True,
            False,
            256,
            1,
            1,
            1,
            True,
            32,
            id="hybrid-tail",
        ),
        pytest.param(
            ("kv_both", "kv_both"),
            True,
            False,
            256,
            2,
            1,
            1,
            False,
            32,
            id="hybrid-tail-tp2",
        ),
    ],
)
def test_shared_cpu_kv_and_weight_version_isolation(
    tmp_path,
    monkeypatch,
    roles,
    save_decode_cache,
    hold_sessions,
    block_size,
    tp,
    dp,
    ep,
    async_scheduling,
    prefix_match_unit,
):
    world_size = tp * dp
    assert torch.cuda.device_count() >= 2 * world_size
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = (
        [device.strip() for device in visible_devices.split(",")]
        if visible_devices is not None
        else [str(i) for i in range(2 * world_size)]
    )
    master = shutil.which("mooncake_master")
    assert master is not None, "install mooncake-transfer-engine"
    port = _free_port()
    config = tmp_path / "store.json"
    config.write_text(
        json.dumps(
            {
                "mode": "embedded",
                "metadata_server": "P2PHANDSHAKE",
                "master_server_address": f"127.0.0.1:{port}",
                "global_segment_size": "1GB",
                "local_buffer_size": "64MB",
                "protocol": "rdma",
                "device_name": "",
                "enable_offload": False,
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config))
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    context = multiprocessing.get_context("spawn")
    namespace = f"titan-test-{uuid.uuid4().hex}"
    processes = []
    with (tmp_path / "master.log").open("w") as log, subprocess.Popen(
        [master, f"--rpc_port={port}", "--rpc_thread_num=4"],
        stdout=log,
        stderr=log,
    ) as server:
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    assert server.poll() is None
                    assert time.monotonic() < deadline
                    time.sleep(0.1)
            queues = []
            block_sizes = []
            for gpu in range(2):
                extra = {"cache_prefix": namespace}
                if save_decode_cache is not None:
                    extra["save_decode_cache"] = save_decode_cache
                connector_args = {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": roles[gpu],
                    "kv_connector_extra_config": extra,
                }
                rank_queues = []
                instance_port = _free_port()
                for rank in range(world_size):
                    commands, results = context.Queue(), context.Queue()
                    process = context.Process(
                        target=_generator_worker,
                        args=(
                            ",".join(
                                device_ids[gpu * world_size : (gpu + 1) * world_size]
                            ),
                            instance_port,
                            connector_args,
                            hold_sessions,
                            block_size,
                            commands,
                            results,
                            rank,
                            tp,
                            dp,
                            ep,
                            async_scheduling,
                            prefix_match_unit,
                        ),
                    )
                    process.start()
                    processes.append(process)
                    rank_queues.append((commands, results))
                queues.append(rank_queues)
                for rank, (_, results) in enumerate(rank_queues):
                    kind, payload = results.get(timeout=300)
                    assert kind == "ready", payload
                    resolved_block_size, device_uuid = payload
                    assert device_uuid == str(
                        torch.cuda.get_device_properties(gpu * world_size + rank).uuid
                    )
                    if block_size is not None and prefix_match_unit is None:
                        assert resolved_block_size == block_size
                    block_sizes.append(resolved_block_size)

            assert block_sizes[0] == block_sizes[1]
            cache_block_size = block_sizes[0]

            def generate(
                gpu,
                version,
                prompt,
                clear_local=False,
                target_dp=0,
                background_prompt=None,
                force_cold=False,
            ):
                request_id = uuid.uuid4().hex
                prompts = {target_dp: prompt}
                if background_prompt is not None:
                    prompts[1 - target_dp] = background_prompt
                for rank, (commands, _) in enumerate(queues[gpu]):
                    dp_rank = rank // tp
                    commands.put(
                        (
                            version,
                            prompts.get(dp_rank),
                            clear_local,
                            f"{request_id}-dp{dp_rank}",
                            (
                                1
                                if prefix_match_unit is not None
                                else max(48, cache_block_size + 2)
                            )
                            + (64 if dp_rank != target_dp else 0),
                            force_cold,
                        )
                    )
                outputs = {}
                for rank, (_, results) in enumerate(queues[gpu]):
                    dp_rank = rank // tp
                    kind, payload = results.get(timeout=90)
                    assert kind == ("output" if dp_rank in prompts else "idle"), payload
                    if kind == "output":
                        if dp_rank in outputs:
                            assert payload == outputs[dp_rank]
                        outputs[dp_rank] = payload
                return outputs[target_dp]

            prompt = list(range(100, 100 + max(512, 2 * cache_block_size)))
            if prefix_match_unit is not None:
                # End prefill at a hash boundary inside a physical block, then
                # finish after one sample: the tail can only save on an idle step.
                prompt += [42] * prefix_match_unit
            # Recomputing the last prompt token limits a repeated prompt's hit
            # to the preceding full block.
            cached_prompt_tokens = (
                (len(prompt) - 1) // cache_block_size * cache_block_size
            )
            first = generate(0, 0, prompt)
            assert first[2] == 0
            if prefix_match_unit is not None:
                assert first[3] > 0, "must exercise a finish-time save without forward"

            # B has never seen this prefix: its first hit must come from A's store.
            next_turn = prompt + first[0] + [42, 43]
            continued = generate(1, 0, next_turn, target_dp=dp - 1)
            if prefix_match_unit is not None:
                assert continued[2] >= len(
                    prompt
                ), "must reload the partial tail from CPU"
            elif save_decode_cache:
                assert continued[2] >= len(prompt) + cache_block_size
            else:
                assert cached_prompt_tokens <= continued[2] <= len(prompt)

            # Compare the continuation that actually consumes decode KV / hybrid
            # tails, not just the original prompt that cannot reuse those states.
            cold = generate(1, 0, next_turn, target_dp=dp - 1, force_cold=True)
            assert cold[2] == 0
            assert continued[0] == cold[0]
            torch.testing.assert_close(
                torch.tensor(continued[1]), torch.tensor(cold[1]), atol=0.02, rtol=0
            )
            shared = generate(1, 0, prompt, clear_local=True, target_dp=dp - 1)
            assert shared[2] >= cached_prompt_tokens
            assert shared[0] == first[0]
            torch.testing.assert_close(
                torch.tensor(shared[1]), torch.tensor(first[1]), atol=0.02, rtol=0
            )

            updated = generate(1, 1, prompt, target_dp=dp - 1)
            assert updated[2] == 0, "old policy KV must not match after a weight update"
            assert updated[1] != first[1], "the test must actually change model weights"
            # Updating B must not purge A's still-valid version from the shared store.
            old = generate(0, 0, prompt, clear_local=True)
            assert old[2] >= cached_prompt_tokens
            assert old[0] == first[0]
            print(
                f"Mooncake cached tokens: cold={first[2]}, shared={shared[2]}, "
                f"next_turn={continued[2]}, new_policy={updated[2]}, old_policy={old[2]}"
            )
            if dp > 1:
                # Both DP replicas work concurrently, then DP0 drains outputs and
                # saves while DP1 is still decoding through EP collectives.
                generate(0, 0, [51] + prompt, background_prompt=[52] + prompt)
            for rank_queues in queues:
                for commands, _ in rank_queues:
                    commands.put(None)
            for process in processes:
                process.join(timeout=30)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
            server.terminate()
            server.wait(timeout=10)
