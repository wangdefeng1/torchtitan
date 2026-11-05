# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Session release must preserve shared prefixes and pending transfer sources."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from torchtitan.rl.generator.vllm.cpu_kv_offload import SimpleCPUOffloadConnector
from torchtitan.rl.generator.vllm.session_kv_offload import SessionKVOffload
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id


def test_release_waits_for_decode_and_saves_and_preserves_shared_prefix():
    connector = Mock()
    connector.get_pending_save_ids.return_value = {0}
    connector.evict_blocks.return_value = None
    first = SimpleNamespace(request_id="first", block_hashes=[b"shared"])
    scheduler = SimpleNamespace(requests={"first": first}, connector=connector)
    sessions = SessionKVOffload(scheduler)
    sessions.track(first, "session-0", 0)
    sessions.release(["session-0"], [])
    connector.evict_blocks.assert_not_called()

    # Final decode hashes appear after release; the request is then removed.
    first.block_hashes.append(b"decode")
    del scheduler.requests["first"]
    sessions.after_step()
    assert sessions.has_pending_work
    connector.evict_blocks.assert_not_called()

    # Another session reuses the prefix while the old save is still in flight.
    second = SimpleNamespace(request_id="second", block_hashes=[b"shared"])
    scheduler.requests["second"] = second
    sessions.track(second, "session-1", 1)
    # A newer unrelated save must not keep this release waiting.
    connector.get_pending_save_ids.return_value = {1}
    sessions.after_step()
    connector.evict_blocks.assert_called_once_with({b"decode"}, {b"shared"})
    assert not sessions.has_pending_work

    # Group release reaches idle sessions too. The final owner can release it.
    del scheduler.requests["second"]
    sessions.after_step()
    connector.get_pending_save_ids.return_value = set()
    sessions.release([], [1])
    assert connector.evict_blocks.call_args.args == ({b"shared"}, set())
    assert not sessions.has_pending_work
    sessions.release(["session-0", "unknown"], [0, 1])
    assert connector.evict_blocks.call_count == 2


def test_eviction_future_keeps_idle_engine_work_alive():
    eviction = Future()
    connector = Mock()
    connector.get_pending_save_ids.return_value = set()
    connector.evict_blocks.return_value = eviction
    request = SimpleNamespace(request_id="r", block_hashes=[b"prefix"])
    scheduler = SimpleNamespace(requests={"r": request}, connector=connector)
    sessions = SessionKVOffload(scheduler)
    sessions.track(request, "session", 0)
    scheduler.requests.clear()
    sessions.release(["session"], [])
    assert sessions.has_pending_work
    sessions.after_step()
    connector.evict_blocks.assert_called_once()
    eviction.set_result(None)
    sessions.after_step()
    assert not sessions.has_pending_work


def test_cpu_eviction_preserves_active_primary_and_secondary_hashes():
    pool = BlockPool(6, True, 16)
    active_primary, active_secondary, finished, duplicate, pinned = pool.get_new_blocks(
        5
    )
    for block, hashes in (
        (active_primary, [b"active-primary", b"released-alias"]),
        (active_secondary, [b"released-primary", b"active-alias"]),
        (finished, [b"finished"]),
        (duplicate, [b"finished"]),
        (pinned, [b"pinned"]),
    ):
        for block_hash in hashes:
            pool._insert_block_hash(
                make_block_hash_with_group_id(block_hash, 0), block, num_tokens=16
            )
    pool.free_blocks([active_primary, active_secondary, finished, duplicate])
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.scheduler_manager = SimpleNamespace(
        cpu_block_pool=pool, group_block_sizes=[16]
    )
    connector.evict_blocks(
        {b"released-alias", b"released-primary", b"finished", b"pinned"},
        {b"active-primary", b"active-alias"},
    )
    # Demotion preserves cached data; allocation actually reclaims it later.
    assert pool.get_cached_block(b"finished", [0]) == [finished]
    assert pool.get_new_blocks(4) == [
        finished,
        duplicate,
        active_primary,
        active_secondary,
    ]
    assert pinned.ref_cnt == 1


def test_cpu_save_snapshot_includes_unsubmitted_tail():
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    manager = SimpleNamespace(
        _store_event_to_blocks={3: object()},
        _abandoned_store_event_to_blocks={2: object()},
        _pending_finished_stores=[object()],
        _store_event_counter=4,
    )
    connector.scheduler_manager = manager
    pending = connector.get_pending_save_ids()
    assert pending == {2, 3, 4}
    manager._pending_finished_stores.clear()
    manager._store_event_to_blocks[4] = object()
    manager._store_event_counter = 5
    assert connector.get_pending_save_ids() == pending
    manager._abandoned_store_event_to_blocks.clear()
    manager._store_event_to_blocks = {5: object()}
    assert not pending & connector.get_pending_save_ids()
