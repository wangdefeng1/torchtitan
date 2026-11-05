# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise skewed KV events across real Gloo ranks without GPUs or a Store."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.rl.generator.vllm import cpu_kv_offload, mooncake
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.generator.vllm.worker import TorchTitanGPUModelRunner
from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreWorker,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata
from vllm.v1.worker import kv_connector_model_runner_mixin


def test_cpu_store_waits_for_tp_workers_but_not_other_dp_replicas():
    manager = SimpleCPUOffloadScheduler.__new__(SimpleCPUOffloadScheduler)
    manager._expected_worker_count = 4  # TP=2, DP=2 external-launcher world.
    manager._store_event_pending_counts = {}
    manager._process_store_event = Mock()

    def init(connector, *_):
        connector.scheduler_manager = manager

    with (
        patch.object(
            cpu_kv_offload.VllmSimpleCPUOffloadConnector, "__init__", new=init
        ),
        patch.object(
            cpu_kv_offload, "get_tp_group", return_value=SimpleNamespace(world_size=2)
        ),
    ):
        connector = cpu_kv_offload.SimpleCPUOffloadConnector(Mock(), Mock(), Mock())
    aggregator = KVOutputAggregator(2)
    for step in range(3):
        outputs = [
            ModelRunnerOutput.with_kv_conn_output_only(
                KVConnectorOutput(
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                        {7: 1}
                        if (rank == 0 and step == 0 or rank == 1 and step == 2)
                        else {}
                    )
                )
            )
            for rank in range(2)
        ]
        connector.update_connector_output(
            aggregator.aggregate(outputs).kv_connector_output
        )
        if step < 2:
            manager._process_store_event.assert_not_called()
    manager._process_store_event.assert_called_once_with(7)


def test_cpu_receive_cleanup_preserves_lru_order_across_local_ids():
    pool = BlockPool(5, True, 16)
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        pool._insert_block_hash(
            block.block_id.to_bytes(8, "little"), block, num_tokens=16
        )
    scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
        cpu_kv_offload.KVOffloadScheduler
    )
    scheduler.logical_request_ids = {"local-z": "first", "local-a": "second"}
    scheduler.requests = {
        key: SimpleNamespace(status=RequestStatus.WAITING_FOR_REMOTE_KVS)
        for key in scheduler.logical_request_ids
    }
    scheduler.finished_recving_kv_req_ids = set()
    assigned = dict(zip(scheduler.logical_request_ids, blocks))

    def complete(output):
        for local_id in output.finished_recving or ():
            pool.free_blocks([assigned[local_id]])

    scheduler.connector = Mock(update_connector_output=Mock(side_effect=complete))
    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={"local-a", "local-z"})
    )
    # Unallocated blocks precede the two released blocks; their order must be
    # logical-ID order even though the local ID order is reversed.
    assert pool.get_new_blocks(4)[-2:] == blocks


def test_no_forward_submits_tail_once_and_polls_delayed_completion():
    # Exercise real Mooncake submission/release code without CUDA or a Store.
    # A finish-time job must be submitted even though no model forward runs.
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MooncakeStoreConnector", kv_role="kv_both"
    )
    worker = MooncakeStoreWorker.__new__(MooncakeStoreWorker)
    worker._capacity_only = False
    worker.can_put = True
    worker.kv_send_thread = Mock()
    connector = Mock(spec=mooncake.VllmMooncakeStoreConnector)
    connector.wait_for_save.side_effect = lambda: worker.wait_for_save(
        connector.bind_connector_metadata.call_args.args[0]
    )
    connector.get_transfer_results.return_value = SimpleNamespace(
        finished_sending=set(), finished_recving={"loaded"}, failed_recving=set()
    )
    connector.get_block_ids_with_load_errors.return_value = set()
    connector.get_kv_connector_stats.return_value = None
    connector.get_kv_connector_kv_cache_events.return_value = None
    connector.build_connector_worker_meta.side_effect = [
        MooncakeStoreWorkerMetadata(),
        MooncakeStoreWorkerMetadata(completed_saves={7: 1}),
    ]
    pool = BlockPool(4, enable_caching=False, hash_block_size=16)
    blocks = pool.get_new_blocks(1)
    store = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
    store._gpu_block_pool = pool
    store._pinned_saves = {7: ([blocks[0].block_id], 1)}
    metadata = MooncakeStoreConnectorMetadata(set(), set())
    tail = ReqMeta(
        req_id="finished",
        token_len_chunk=0,
        block_ids=([blocks[0].block_id],),
        block_hashes=[],
        can_save=True,
        store_job_id=7,
        boundary_state_offloads=[(0, blocks[0].block_id, 24)],
    )
    metadata.add_request(tail)
    output = SchedulerOutput.make_empty()
    output.kv_connector_metadata = metadata
    with (
        patch.object(
            kv_connector_model_runner_mixin,
            "get_kv_transfer_group",
            return_value=connector,
        ),
        patch.object(
            kv_connector_model_runner_mixin, "has_kv_transfer_group", return_value=True
        ),
        patch("torch.cuda.Event") as event,
    ):
        first = TorchTitanGPUModelRunner.kv_connector_no_forward(output, config)
        worker.kv_send_thread.add_request.assert_called_once_with(tail)
        event.return_value.record.assert_called_once()
        assert tail.current_event is event.return_value
        assert first.kv_connector_output.finished_recving == {"loaded"}
        store.update_connector_output(first.kv_connector_output)
        assert blocks[0].ref_cnt == 1 and store.has_pending_push_work()

        # Later empty steps only poll; they must not resubmit the old metadata.
        output.kv_connector_metadata = MooncakeStoreConnectorMetadata(set(), set())
        second = TorchTitanGPUModelRunner.kv_connector_no_forward(output, config)
        worker.kv_send_thread.add_request.assert_called_once_with(tail)
        store.update_connector_output(second.kv_connector_output)
        assert blocks[0].ref_cnt == 0 and not store.has_pending_push_work()
        assert connector.clear_connector_metadata.call_count == 2


def _coordinate_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        ranks=[0, 1],
        cpu_group=dist.group.WORLD,
    )
    try:
        with (
            patch.object(mooncake, "get_tp_group", return_value=group),
            patch.object(cpu_kv_offload, "get_tp_group", return_value=group),
        ):
            # A follower must never query independently. Pending and hit results
            # are both shared, including the load metadata consumed by allocation.
            connector = mooncake.MooncakeStoreConnector.__new__(
                mooncake.MooncakeStoreConnector
            )
            connector.connector_worker = None
            spec = LoadSpec(16, 48, False)
            connector.connector_scheduler = SimpleNamespace(load_specs={})
            request = SimpleNamespace(request_id=f"lookup-rank{rank}")

            def lookup(*args):
                assert rank == 0
                connector.connector_scheduler.load_specs[request.request_id] = spec
                return 32, True

            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                side_effect=lookup,
            ):
                assert connector.get_num_new_matched_tokens(request, 16) == (32, True)
                assert (
                    connector.connector_scheduler.load_specs[request.request_id] == spec
                )
            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                return_value=(None, False),
            ) as pending:
                assert connector.get_num_new_matched_tokens(request, 16) == (
                    None,
                    False,
                )
                assert pending.call_count == (1 if rank == 0 else 0)
            with patch.object(
                mooncake.VllmMooncakeStoreConnector,
                "get_num_new_matched_tokens",
                side_effect=RuntimeError("lookup unavailable"),
            ):
                with pytest.raises(RuntimeError, match="lookup unavailable"):
                    connector.get_num_new_matched_tokens(request, 16)

            # One shared work flag covers requests, pending KV and queued outputs.
            # A locally idle rank must keep stepping until every rank has drained.
            engine = VLLMEngine.__new__(VLLMEngine)
            engine._session_kv_offload = None
            engine._coordinate_kv_offload = True
            engine._uses_kv_offload = True
            engine._loaded_policy_version = 0
            engine._reset_kv_cache_on_model_state_dict_load = False
            model = SimpleNamespace(
                load_state_dict=Mock(), _prefetched_model_state_dict={}
            )
            engine._get_model = lambda: model
            state = SimpleNamespace(requests=False, scheduled=False)
            core = SimpleNamespace(
                scheduler=SimpleNamespace(has_requests=lambda: state.scheduled),
                batch_queue=None,
            )
            engine._engine = SimpleNamespace(
                output_processor=SimpleNamespace(
                    has_unfinished_requests=lambda: state.requests
                ),
                engine_core=SimpleNamespace(engine_core=core),
            )
            with patch(
                "torchtitan.rl.generator.vllm.engine.get_world_group",
                return_value=group,
            ):
                for source in ("requests", "scheduler", "queue", "idle"):
                    state.requests = rank == 1 and source == "requests"
                    state.scheduled = rank == 1 and source == "scheduler"
                    core.batch_queue = (
                        [object()] if rank == 1 and source == "queue" else None
                    )
                    assert engine._sync_unfinished_requests() == (source != "idle")
                    assert engine.has_unfinished_requests() == (source != "idle")

                    model.load_state_dict.reset_mock()
                    if source == "requests":
                        # Only rank 1 is active, but both must reject before
                        # changing weights; the work flag alone is insufficient.
                        with pytest.raises(RuntimeError, match="hot_swap=False"):
                            engine.load_prefetched_model_state_dict(1)
                        model.load_state_dict.assert_not_called()
                        assert engine._loaded_policy_version == 0
                    else:
                        # Saving old, versioned KV does not block a drained load.
                        engine.load_prefetched_model_state_dict(1)
                        model.load_state_dict.assert_called_once_with({})
                        assert engine._loaded_policy_version == 1

                    # Single-rank generators use the same combined local condition.
                    engine._coordinate_kv_offload = False
                    assert engine.has_unfinished_requests() == (
                        rank == 1 and source != "idle"
                    )
                    engine._coordinate_kv_offload = True

            # Use upstream block accounting: the first save completion must not
            # release references, and a failed receive must wait for the other
            # worker before the scheduler can decide to retry/recompute.
            pool = BlockPool(8, False, 16)
            blocks = pool.get_new_blocks(2)
            store_scheduler = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
            store_scheduler._gpu_block_pool = pool
            store_scheduler._pinned_saves = {7: ([b.block_id for b in blocks], 2)}
            scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
                cpu_kv_offload.KVOffloadScheduler
            )
            scheduler._kv_output_aggregator = KVOutputAggregator(2)
            local_ids = {
                logical: f"{logical}-rank{rank}"
                for logical in ("load", "failed", "send")
            }
            scheduler.requests = dict.fromkeys(local_ids.values())
            scheduler.logical_request_ids = {
                local: logical for logical, local in local_ids.items()
            }
            scheduler.local_request_ids = local_ids.copy()
            observed = []

            def apply_output(_self, _scheduled, output):
                kv = output.kv_connector_output
                store_scheduler.update_connector_output(kv)
                observed.append(kv)
                # Sampling output stays local; only control metadata is gathered.
                assert output.sampled_token_ids == [[rank]]
                assert output.req_ids == [local_ids["load"]]
                return {}

            with patch.object(
                cpu_kv_offload.Scheduler, "update_from_output", new=apply_output
            ):
                for step in range(3):
                    report = rank == 0 and step == 0 or rank == 1 and step == 2
                    kv = KVConnectorOutput(
                        finished_recving=(
                            {local_ids["load"], local_ids["failed"]} if report else None
                        ),
                        finished_sending={local_ids["send"]} if report else None,
                        failed_recving=(
                            {local_ids["failed"]} if rank == 0 and step == 0 else set()
                        ),
                        kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                            {7: 1} if report else {}
                        ),
                    )
                    output = ModelRunnerOutput(
                        req_ids=[local_ids["load"]],
                        req_id_to_index={local_ids["load"]: 0},
                        sampled_token_ids=[[rank]],
                        kv_connector_output=kv,
                    )
                    scheduler.update_from_output(None, output)
                    if step < 2:
                        assert not observed[-1].finished_recving
                        assert not observed[-1].finished_sending
                        assert not observed[-1].failed_recving
                        assert all(b.ref_cnt == 1 for b in blocks)
                        assert store_scheduler.has_pending_push_work()
                    else:
                        assert observed[-1].finished_recving == {
                            local_ids["load"],
                            local_ids["failed"],
                        }
                        assert observed[-1].finished_sending == {local_ids["send"]}
                        assert observed[-1].failed_recving == {local_ids["failed"]}
                        assert all(b.ref_cnt == 0 for b in blocks)
                        assert not store_scheduler.has_pending_push_work()
    finally:
        dist.destroy_process_group()


def test_lookup_and_delayed_completion_coordination(tmp_path):
    mp.spawn(_coordinate_worker, args=(f"file://{tmp_path}/rendezvous",), nprocs=2)


def test_idle_dp_forward_does_not_skip_queued_output_drain():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._coordinate_kv_offload = True
    engine._session_kv_holder = None
    engine._session_kv_offload = None
    calls = []

    def coordinate(has_requests):
        assert not has_requests
        engine._engine.should_execute_dummy_batch = True
        calls.append("coordinate")

    def drain():
        assert not engine._engine.should_execute_dummy_batch
        calls.append("drain")
        return []

    engine._engine = SimpleNamespace(
        dp_group=object(),
        has_unfinished_requests_dp=coordinate,
        engine_core=SimpleNamespace(
            engine_core=SimpleNamespace(
                scheduler=SimpleNamespace(has_requests=lambda: False),
                batch_queue=[object()],
            ),
            execute_dummy_batch=lambda: calls.append("dummy"),
        ),
        step=drain,
    )
    assert engine.step() == []
    assert calls == ["coordinate", "dummy", "drain"]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_scheduler_uses_resolved_async_setting(async_scheduling):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)
    )
    with (
        patch.object(cpu_kv_offload, "KVOffloadScheduler") as sync,
        patch.object(cpu_kv_offload, "AsyncKVOffloadScheduler") as async_,
    ):
        result = cpu_kv_offload.create_kv_offload_scheduler(config)
        selected, other = (async_, sync) if async_scheduling else (sync, async_)
        assert result is selected.return_value
        selected.assert_called_once_with(config)
        other.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_each_dp_replica_has_own_lookup_server_and_save_quorum(rank):
    worker = SimpleNamespace(lookup_server=None)
    scheduler = SimpleNamespace(_num_workers=4)

    def init(connector, *args):
        connector.connector_worker = worker
        connector.connector_scheduler = scheduler

    group = SimpleNamespace(rank_in_group=rank, world_size=2)
    with (
        patch.object(mooncake.VllmMooncakeStoreConnector, "__init__", new=init),
        patch.object(mooncake, "get_tp_group", return_value=group),
        patch.object(mooncake, "LookupKeyServer") as server,
    ):
        connector = mooncake.MooncakeStoreConnector(Mock(), Mock())
        assert connector.connector_scheduler._num_workers == 2
        assert server.call_count == (1 if rank == 0 else 0)
        # Prevent the upstream destructor from calling close on this fake worker.
        connector.connector_worker = None


@pytest.mark.parametrize("suffix", ["00000000", "ffffffff"])
def test_priority_ties_use_logical_ids_across_waiting_queues(suffix):
    # Native vLLM orders these IDs differently depending on the random suffix.
    # A TP leader's lookup must never be applied to another request on a follower.
    scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
        cpu_kv_offload.KVOffloadScheduler
    )
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.logical_request_ids = {}
    scheduler.local_request_ids = {}
    scheduler.requests = {}
    scheduler.connector = None
    scheduler.log_stats = False
    scheduler.spec_decode_metrics_level = "none"
    scheduler.waiting = cpu_kv_offload._LogicalPriorityRequestQueue(
        scheduler.logical_request_ids
    )
    scheduler.skipped_waiting = cpu_kv_offload._LogicalPriorityRequestQueue(
        scheduler.logical_request_ids
    )
    requests = []
    for logical_id in ("request-5", "request"):
        request = Request(
            f"{logical_id}-{suffix}",
            [1],
            SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": {"titan_request_id": logical_id}},
            ),
            None,
            arrival_time=1.0,
        )
        scheduler.add_request(request)
        requests.append(request)
    earlier = requests[1]
    assert scheduler.waiting.pop_request() is earlier
    # The same tie-break must hold between the ready and blocked queue heads.
    earlier.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.skipped_waiting.add_request(earlier)
    assert scheduler._select_waiting_queue_for_scheduling() is scheduler.skipped_waiting
    scheduler.waiting.prepend_requests(scheduler.skipped_waiting)
    scheduler.skipped_waiting.remove_request(earlier)
    assert list(scheduler.waiting) == [earlier, requests[0]]
    scheduler.waiting.remove_requests([earlier])
    scheduler.waiting.prepend_request(earlier)
    assert scheduler.waiting.pop_request() is earlier
    assert scheduler.waiting.pop_request() is requests[0]
    assert not scheduler.waiting


@pytest.mark.parametrize("late_receive", [False, True])
@pytest.mark.parametrize(
    "finished_status", [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR]
)
def test_terminal_cleanup_preserves_block_identity(late_receive, finished_status):
    # Use real queues and KV allocation/freeing without constructing a GPU worker.
    # Per-process string hash order must not change a later invalid block's owner.
    scheduler = cpu_kv_offload.KVOffloadScheduler.__new__(
        cpu_kv_offload.KVOffloadScheduler
    )
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.defer_block_free = False
    scheduler._inflight_prefills = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.requests = {}
    scheduler.logical_request_ids = {}
    scheduler.local_request_ids = {}
    scheduler.spec_decode_metrics_level = "none"
    scheduler.log_stats = False
    scheduler.running = []
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.encoder_cache_manager = EncoderCacheManager(0)
    scheduler.block_size = 16
    manager = scheduler.kv_cache_manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=5,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
                    ),
                )
            ],
        ),
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=False,
    )

    def allocate(request_id, logical_id):
        request = Request(
            request_id,
            [1] * 32,
            SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": {"titan_request_id": logical_id}},
            ),
            None,
        )
        scheduler.add_request(request)
        assert manager.allocate_slots(request, num_new_tokens=32) is not None
        request.num_computed_tokens = 32
        return request

    # Deliberately reverse local-ID ordering: block release must use logical IDs.
    for request_id, logical_id in (("local-B", "request-A"), ("local-A", "request-B")):
        request = allocate(request_id, logical_id)
        scheduler.waiting.remove_request(request)
        if late_receive:
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            scheduler.skipped_waiting.add_request(request)
        else:
            request.status = RequestStatus.RUNNING
            scheduler.running.append(request)
    original_b_blocks = manager.get_block_ids("local-A")[0].copy()
    scheduler.finish_requests({"local-A", "local-B", "unknown"}, finished_status)
    if late_receive:
        assert manager.block_pool.get_num_free_blocks() == 0
        assert len(scheduler.local_request_ids) == 2
        scheduler._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_recving={"local-A", "local-B"})
        )
    assert not scheduler.requests
    assert not scheduler.logical_request_ids and not scheduler.local_request_ids
    assert (
        not scheduler.running
        and not scheduler.waiting
        and not scheduler.skipped_waiting
    )

    # A logical ID can be reused after its old request/receive is fully retired.
    following = [allocate("next-C", "request-A"), allocate("next-D", "request-B")]
    # Non-cached blocks are reused LIFO: canonical A-then-B cleanup puts B first.
    assert manager.get_block_ids("next-C")[0] == original_b_blocks[::-1]
    affected, _, _ = scheduler._update_requests_with_invalid_blocks(
        following, {original_b_blocks[-1]}, {}, evict_blocks=False
    )
    assert affected == {"next-C"}
