"""Embedding must run off the event loop.

Embedders are synchronous (ONNX inference or a blocking HTTP call). If the
service invoked them directly, one embed would stall every concurrent MCP
request; these tests pin the worker-thread offload in place.
"""

from __future__ import annotations

import threading

from hypermnesia.config import Settings
from hypermnesia.service import MemoryService


class RecordingEmbedder:
    """Fake embedder that records which thread each call runs on."""

    model_id = "recording"
    dim = 3

    def __init__(self):
        self.threads: list[threading.Thread] = []

    def embed_documents(self, texts):
        self.threads.append(threading.current_thread())
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.threads.append(threading.current_thread())
        return [0.0, 1.0, 0.0]


def _svc(embedder) -> MemoryService:
    return MemoryService(pool=None, embedder=embedder, settings=Settings())


async def test_embed_record_runs_in_worker_thread():
    emb = RecordingEmbedder()
    vec = await _svc(emb)._embed_record("desc", "content")
    assert vec.to_list() == [1.0, 0.0, 0.0]
    assert emb.threads == [emb.threads[0]]
    assert emb.threads[0] is not threading.current_thread()


async def test_embed_query_runs_in_worker_thread():
    emb = RecordingEmbedder()
    vec = await _svc(emb)._embed_query("what is stored?")
    assert vec.to_list() == [0.0, 1.0, 0.0]
    assert emb.threads[0] is not threading.current_thread()
