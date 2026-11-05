# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Translate session-end notifications into CPU KV cache eviction hints."""

from concurrent.futures import Future
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request


class SessionKVOffload:
    """Track session hashes independently of GPU KV retention.

    Keep request references until scheduler cleanup to capture decode hashes.
    After releases and outstanding saves, ask the connector to evict hashes
    unused by other local sessions. This is a one-shot hint, not a tombstone:
    future requests may cache the same prefix again. Mooncake does not provide
    global session ownership, so remote sessions may lose a cache hit.
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._sessions: dict[str, tuple[int, set[BlockHash]]] = {}
        self._requests: dict[str, tuple[Request, str]] = {}
        self._released_sessions: set[str] = set()
        self._released_hashes: set[BlockHash] = set()
        self._eviction_hashes: set[BlockHash] = set()
        self._pending_saves: set[int] = set()
        self._eviction: Future[None] | None = None

    def track(self, request: Request, session_id: str, group_id: int) -> None:
        group, _ = self._sessions.setdefault(session_id, (group_id, set()))
        assert group == group_id
        self._requests[request.request_id] = (request, session_id)

    def release(self, session_ids: list[str], group_ids: list[int]) -> None:
        """Release sessions, including all sessions of finished rollout groups."""
        groups = set(group_ids)
        self._released_sessions.update(set(session_ids) & self._sessions.keys())
        self._released_sessions.update(
            session for session, (group, _) in self._sessions.items() if group in groups
        )
        self.after_step()

    @property
    def has_pending_work(self) -> bool:
        return bool(
            self._released_sessions
            or self._released_hashes
            or self._eviction_hashes
            or self._eviction is not None
        )

    def after_step(self) -> None:
        """Capture completed requests and apply releases after async saves drain."""
        for request_id, (request, session) in list(self._requests.items()):
            if request_id not in self._scheduler.requests:
                self._sessions[session][1].update(request.block_hashes)
                del self._requests[request_id]

        live_sessions = {session for _, session in self._requests.values()}
        for session in sorted(self._released_sessions - live_sessions):
            _, hashes = self._sessions.pop(session)
            self._released_hashes.update(hashes)
            self._released_sessions.remove(session)

        if self._eviction is not None:
            if not self._eviction.done():
                return
            self._eviction.result()
            self._eviction = None

        connector = self._scheduler.connector
        if not self._eviction_hashes:
            if not self._released_hashes:
                return
            self._eviction_hashes, self._released_hashes = self._released_hashes, set()
            # Include finish-time saves queued for the next step. Freeze this
            # batch so unrelated later saves cannot postpone its release forever.
            self._pending_saves = connector.get_pending_save_ids()
        if self._pending_saves & connector.get_pending_save_ids():
            return

        active_hashes = set().union(*(hashes for _, hashes in self._sessions.values()))
        for request, _ in self._requests.values():
            active_hashes.update(request.block_hashes)
        released = self._eviction_hashes - active_hashes
        self._eviction_hashes.clear()
        self._pending_saves.clear()
        if released:
            self._eviction = connector.evict_blocks(released, active_hashes)
