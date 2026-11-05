# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coordinate external-launcher schedulers while KV transfers run asynchronously.

Each TP group shares request ordering and transfer completions. DP replicas keep
independent caches and requests. Logical Titan IDs match reports across ranks;
each scheduler applies the result using its own vLLM request IDs.

TODO(acisseJZhong): upstream external-launcher KV coordination to remove these
adapters: https://github.com/vllm-project/vllm/issues/11400
"""

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any

import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector as VllmSimpleCPUOffloadConnector,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


class SimpleCPUOffloadConnector(VllmSimpleCPUOffloadConnector):
    """Local CPU caches with a save-completion quorum scoped to one TP group.

    Every scheduler queries its own CPU cache. Coordinated store completions and
    block release keep those caches consistent; no leader lookup is needed.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.scheduler_manager is not None:
            # external_launcher world_size includes independent DP replicas.
            self.scheduler_manager._expected_worker_count = get_tp_group().world_size

    def get_pending_save_ids(self) -> set[int]:
        """Snapshot queued/in-flight saves, including the next finish-time batch."""
        manager = self.scheduler_manager
        assert manager is not None
        pending = set(manager._store_event_to_blocks) | set(
            manager._abandoned_store_event_to_blocks
        )
        if manager._pending_finished_stores:
            # build_connector_meta folds these tails into the next store event.
            pending.add(manager._store_event_counter)
        return pending

    def evict_blocks(
        self, block_hashes: set[BlockHash], active_hashes: set[BlockHash]
    ) -> None:
        """Make unused CPU blocks the next evicted, preserving their cached data."""
        manager = self.scheduler_manager
        assert manager is not None
        pool = manager.cpu_block_pool
        blocks = {}
        for block_hash in sorted(block_hashes):
            for group_id in range(len(manager.group_block_sizes)):
                cached = pool.cached_block_hash_to_block._cache.get(
                    make_block_hash_with_group_id(block_hash, group_id)
                )
                if cached is None:
                    continue
                # Simultaneous saves may cache several physical copies of a hash.
                for block in cached.values() if isinstance(cached, dict) else [cached]:
                    aliases = [
                        block.block_hash,
                        *pool.cached_block_hashes_by_block.get(block.block_id, ()),
                    ]
                    if block.ref_cnt == 0 and not any(
                        get_block_hash(alias) in active_hashes for alias in aliases
                    ):
                        blocks[block.block_id] = block
        # TODO(acisseJZhong): upstream a hash-based eviction-priority API instead
        # of accessing vLLM's cache index and free queue directly:
        # https://github.com/vllm-project/vllm/issues/11400
        for block in blocks.values():
            pool.free_block_queue.remove(block)
        pool.free_block_queue.prepend_n(list(blocks.values()))


class _LogicalPriorityRequestQueue(RequestQueue):
    """A priority heap with a shared logical-ID tie-break instead of local IDs."""

    def __init__(self, logical_ids: dict[str, str]) -> None:
        self._logical_ids = logical_ids
        self._heap: list[tuple[int, float, str, Request]] = []

    def _entry(self, request: Request) -> tuple[int, float, str, Request]:
        return (
            request.priority,
            request.arrival_time,
            self._logical_ids[request.request_id],
            request,
        )

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, self._entry(request))

    def pop_request(self) -> Request:
        return heapq.heappop(self._heap)[-1]

    def peek_request(self) -> Request:
        return self._heap[0][-1]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._heap.remove(self._entry(request))
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        ids = {request.request_id for request in requests}
        self._heap = [entry for entry in self._heap if entry[-1].request_id not in ids]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        return (entry[-1] for entry in sorted(self._heap))


class KVOffloadScheduler(Scheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._kv_output_aggregator = KVOutputAggregator.from_connector(
            self.connector, get_tp_group().world_size
        )
        self.logical_request_ids: dict[str, str] = {}
        self.local_request_ids: dict[str, str] = {}
        if self.policy == SchedulingPolicy.PRIORITY:
            self.waiting = _LogicalPriorityRequestQueue(self.logical_request_ids)
            self.skipped_waiting = _LogicalPriorityRequestQueue(
                self.logical_request_ids
            )

    def add_request(self, request: Request) -> None:
        logical_id = request.kv_transfer_params["titan_request_id"]
        assert logical_id not in self.local_request_ids
        self.logical_request_ids[request.request_id] = logical_id
        self.local_request_ids[logical_id] = request.request_id
        super().add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if (
            self.policy == SchedulingPolicy.PRIORITY
            and self.waiting
            and self.skipped_waiting
        ):

            def key(queue: RequestQueue) -> tuple[int, float, str]:
                request = queue.peek_request()
                return (
                    request.priority,
                    request.arrival_time,
                    self.logical_request_ids[request.request_id],
                )

            return min((self.waiting, self.skipped_waiting), key=key)
        return super()._select_waiting_queue_for_scheduling()

    def _free_blocks(self, request: Request) -> None:
        super()._free_blocks(request)
        # Keep IDs while a cancelled receive still owns the request. Store saves
        # own separate integer job IDs and block references, not request IDs.
        logical_id = self.logical_request_ids.pop(request.request_id)
        del self.local_request_ids[logical_id]

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Aggregate TP transfer reports before updating this rank's scheduler."""
        group = get_tp_group()
        if group.world_size > 1:
            outputs: list[KVConnectorOutput | None] = [None] * group.world_size
            dist.all_gather_object(
                outputs,
                _remap_request_ids(
                    model_runner_output.kv_connector_output, self.logical_request_ids
                ),
                group=group.cpu_group,
            )
            # The upstream aggregator takes ModelRunnerOutputs. Each shallow
            # copy keeps this rank's model output and carries one worker's KV
            # report; only those reports are aggregated, across ranks and steps.
            aggregated_output = self._kv_output_aggregator.aggregate(
                [replace(model_runner_output, kv_connector_output=o) for o in outputs]
            )
            assert aggregated_output is not None
            aggregated_output.kv_connector_output = _remap_request_ids(
                aggregated_output.kv_connector_output, self.local_request_ids
            )
            model_runner_output = aggregated_output
        return super().update_from_output(scheduler_output, model_runner_output)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[Request]:
        # Free in the same order on every TP rank so future physical block IDs
        # agree. Upstream converts iterable IDs back to a set; pass strings.
        if isinstance(request_ids, str):
            return super().finish_requests(request_ids, finished_status)
        ids = self.requests if request_ids is None else request_ids
        finished = []
        for request_id in sorted(
            set(ids) & self.requests.keys(),
            key=lambda local_id: self.logical_request_ids[local_id],
        ):
            finished.extend(super().finish_requests(request_id, finished_status))
        return finished

    def _update_from_kv_xfer_finished(self, output: KVConnectorOutput) -> None:
        """Apply aggregated KV completions using each rank's local request IDs.

        update_from_output() has already synchronized the reports. Unlike the
        base scheduler, process completions in shared Titan ID order so block
        reuse stays consistent across TP ranks.
        """
        # Store tracks save jobs by job ID and releases their pinned blocks once
        # all required workers finish, independently of request cleanup below.
        if self.connector is not None:
            self.connector.update_connector_output(
                replace(output, finished_recving=None)
            )
        for request_id in sorted(
            output.finished_recving or (),
            key=lambda local_id: self.logical_request_ids[local_id],
        ):
            # CPU offload also frees cache references in this callback. Apply
            # receives in logical order so every rank keeps the same LRU order.
            if self.connector is not None:
                self.connector.update_connector_output(
                    KVConnectorOutput(finished_recving={request_id})
                )
            request = self.requests[request_id]
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # schedule() checks this flag before resuming the request.
                self.finished_recving_kv_req_ids.add(request_id)
            else:
                # The request ended during the load; its blocks can now be freed.
                assert RequestStatus.is_finished(request.status)
                self._free_blocks(request)

        # Store saves use job metadata above; retain generic send-completion handling.
        for request_id in sorted(
            output.finished_sending or (),
            key=lambda local_id: self.logical_request_ids[local_id],
        ):
            self._free_blocks(self.requests[request_id])


def _remap_request_ids(
    output: KVConnectorOutput | None, request_ids: dict[str, str]
) -> KVConnectorOutput | None:
    """Translate request IDs in reports, leaving Store save-job IDs unchanged."""
    if output is None:
        return None
    return replace(
        output,
        finished_recving=(
            {request_ids[req_id] for req_id in output.finished_recving}
            if output.finished_recving is not None
            else None
        ),
        finished_sending=(
            {request_ids[req_id] for req_id in output.finished_sending}
            if output.finished_sending is not None
            else None
        ),
        failed_recving={request_ids[req_id] for req_id in output.failed_recving},
    )


class AsyncKVOffloadScheduler(AsyncScheduler, KVOffloadScheduler):
    """Preserve vLLM's async scheduling while coordinating KV completions."""


def create_kv_offload_scheduler(
    vllm_config: VllmConfig,
    *args: Any,
    **kwargs: Any,
) -> Scheduler:
    # vLLM resolves automatic async scheduling while building VllmConfig, after
    # Titan creates EngineArgs. Select the matching implementation here.
    scheduler_cls = (
        AsyncKVOffloadScheduler
        if vllm_config.scheduler_config.async_scheduling
        else KVOffloadScheduler
    )
    return scheduler_cls(vllm_config, *args, **kwargs)
