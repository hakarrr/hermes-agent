"""Task metadata describes state changes, not polling or stream replay."""
from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import protocol


@pytest.mark.parametrize("state", [protocol.STATE_COMPLETED, protocol.STATE_FAILED,
                                  protocol.STATE_INPUT_REQUIRED, protocol.STATE_CANCELED,
                                  protocol.STATE_REJECTED])
def test_repeated_reads_and_list_preserve_metadata(monkeypatch, state):
    clock = iter(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"])
    monkeypatch.setattr(protocol, "now_iso", lambda: next(clock))
    store = protocol.TaskStore()
    store.create("task", "context", "peer")
    store.complete("task", state, "answer")
    monkeypatch.setattr(protocol, "now_iso", lambda: "2099-01-01T00:00:00Z")
    first = store.to_task(store.get("task"))
    assert first["status"]["timestamp"] == "2026-01-01T00:00:01Z"
    assert first == store.to_task(store.get("task"))
    records, _ = store.list()
    assert first == store.to_task(records[0])
    without_artifacts = store.to_task(records[0], include_artifacts=False)
    assert without_artifacts == {key: value for key, value in first.items() if key != "artifacts"}


def test_timestamp_changes_only_on_state_transition(monkeypatch):
    monkeypatch.setattr(protocol, "now_iso", lambda: "2026-01-01T00:00:00Z")
    store = protocol.TaskStore()
    store.create("task", "context", "peer")
    monkeypatch.setattr(protocol, "now_iso", lambda: "2026-01-01T00:00:01Z")
    assert store.to_task(store.get("task"))["status"]["timestamp"] == "2026-01-01T00:00:00Z"
    store.set_state("task", protocol.STATE_WORKING)
    working = store.to_task(store.get("task"))
    assert working["status"]["timestamp"] == "2026-01-01T00:00:01Z"
    monkeypatch.setattr(protocol, "now_iso", lambda: "2026-01-01T00:00:02Z")
    store.set_state("task", protocol.STATE_WORKING)
    assert store.to_task(store.get("task")) == working
    store.complete("task", protocol.STATE_COMPLETED, "answer")
    terminal = store.to_task(store.get("task"))
    monkeypatch.setattr(protocol, "now_iso", lambda: "2099-01-01T00:00:00Z")
    store.set_state("task", protocol.STATE_WORKING)
    assert store.complete("task", protocol.STATE_FAILED, "late") is None
    assert store.to_task(store.get("task")) == terminal


def _adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter
    return A2AAdapter(PlatformConfig(enabled=True))


@pytest.mark.parametrize("state", [protocol.STATE_COMPLETED, protocol.STATE_FAILED,
                                  protocol.STATE_INPUT_REQUIRED])
def test_terminal_stream_replay_matches_get(monkeypatch, state):
    adapter = _adapter()
    adapter.tasks.create("task", "context", "peer")
    adapter.tasks.complete("task", state, "answer")
    expected = adapter.tasks.to_task(adapter.tasks.get("task"))
    monkeypatch.setattr(protocol, "now_iso", lambda: "2099-01-01T00:00:00Z")
    frames = []
    for _ in range(2):
        handler = SimpleNamespace(wfile=io.BytesIO())
        adapter._emit_terminal(handler, "task", "context", state, "answer", req_id=1)
        frames.append([json.loads(line[6:])["result"] for line in handler.wfile.getvalue().decode().splitlines()
                       if line.startswith("data: ")])
    assert frames[0] == frames[1]
    status = frames[0][-1]["statusUpdate"]["status"]
    assert status["timestamp"] == expected["status"]["timestamp"]
    if state == protocol.STATE_COMPLETED:
        assert frames[0][0]["artifactUpdate"]["artifact"] == expected["artifacts"][0]
    else:
        assert status == expected["status"]


def test_send_response_matches_later_get(monkeypatch):
    adapter = _adapter()
    rec = adapter.tasks.create("task", "context", "peer")
    pending = {**rec, "future": None}
    monkeypatch.setattr(adapter, "_prepare_task", lambda *args, **kwargs: (None, pending))
    monkeypatch.setattr(adapter, "_await_reply", lambda pending: (protocol.STATE_COMPLETED, "answer"))

    def finalize(pending, state, reply):
        adapter.tasks.complete("task", state, reply)
        return state, reply

    monkeypatch.setattr(adapter, "_finalize_task", finalize)
    initial = adapter._rpc_message_send(1, {}, "peer")["result"]
    monkeypatch.setattr(protocol, "now_iso", lambda: "2099-01-01T00:00:00Z")
    assert initial == adapter._rpc_tasks_get(2, {"taskId": "task"})["result"]


def test_distinct_tasks_and_new_replies_have_distinct_ids():
    store = protocol.TaskStore()
    for task_id in ("first", "second"):
        store.create(task_id, "context", "peer")
        store.complete(task_id, protocol.STATE_INPUT_REQUIRED, "question")
    first = store.to_task(store.get("first"))
    second = store.to_task(store.get("second"))
    assert first["status"]["message"]["messageId"] != second["status"]["message"]["messageId"]
    store.complete("first", protocol.STATE_COMPLETED, "answer")
    completed = store.to_task(store.get("first"))
    assert completed["status"]["message"]["messageId"] != first["status"]["message"]["messageId"]
    store.complete("second", protocol.STATE_COMPLETED, "answer")
    assert completed["artifacts"][0]["artifactId"] != store.to_task(store.get("second"))["artifacts"][0]["artifactId"]


def test_render_does_not_generate_metadata(monkeypatch):
    store = protocol.TaskStore()
    store.create("task", "context", "peer")
    store.complete("task", protocol.STATE_COMPLETED, "answer")

    def unexpected():
        raise AssertionError("rendering must not generate time or IDs")

    monkeypatch.setattr(protocol, "now_iso", unexpected)
    monkeypatch.setattr(protocol.uuid, "uuid4", unexpected)
    assert store.to_task(store.get("task"))["artifacts"]


def test_push_uses_stored_status_and_keeps_reply_limit(monkeypatch):
    from plugins.platforms.a2a import security
    adapter = _adapter()
    reply = "x" * 2100
    adapter.tasks.create("task", "context", "peer")
    adapter.tasks.complete("task", protocol.STATE_COMPLETED, reply)
    expected = adapter.tasks.to_task(adapter.tasks.get("task"))["status"]
    adapter.tasks.set_push_config("task", "https://callback.example/result")
    monkeypatch.setattr(security, "is_safe_callback_url", lambda *args, **kwargs: True)
    sent = []
    monkeypatch.setattr(security, "post_callback", lambda url, body, *args, **kwargs: sent.append(json.loads(body)) or 200)
    monkeypatch.setattr(protocol, "now_iso", lambda: "2099-01-01T00:00:00Z")
    adapter._send_push_notification("task", "context", reply, protocol.STATE_COMPLETED)
    actual = sent[0]["statusUpdate"]["status"]
    assert actual["timestamp"] == expected["timestamp"]
    assert actual["message"]["messageId"] == expected["message"]["messageId"]
    assert protocol.extract_text(actual["message"]) == "x" * 2000
    assert protocol.extract_text(adapter.tasks.to_task(adapter.tasks.get("task"))["status"]["message"]) == reply


@pytest.mark.integration
@pytest.mark.parametrize("streaming", [False, True])
def test_live_transport_metadata_survives_get_list_and_subscribe(monkeypatch, streaming):
    from tests.plugins.test_a2a_plugin import _make_live_adapter, _post_json, _send_body
    from tests.plugins.test_a2a_phase23 import _post_sse

    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    adapter, base = _make_live_adapter(monkeypatch, reply_fn=lambda event: "answer")

    async def run():
        assert await adapter.connect()
        try:
            if streaming:
                body = _send_body("hello")
                body["method"] = "message/stream"
                initial, _ = await asyncio.to_thread(_post_sse, base + "/", body)
                task_id = initial[0]["task"]["id"]
                sent_artifact = next(frame["artifactUpdate"]["artifact"] for frame in initial if "artifactUpdate" in frame)
                sent_status = initial[-1]["statusUpdate"]["status"]
            else:
                response = await asyncio.to_thread(_post_json, base + "/", _send_body("hello"))
                task = response["result"]
                task_id = task["id"]
                sent_artifact = task["artifacts"][0]
                sent_status = task["status"]

            async def rpc(method, params):
                return (await asyncio.to_thread(_post_json, base + "/", {
                    "jsonrpc": "2.0", "id": "read", "method": method, "params": params}))["result"]

            fetched = await rpc("GetTask", {"taskId": task_id})
            assert fetched == await rpc("GetTask", {"taskId": task_id})
            assert fetched == (await rpc("ListTasks", {"includeArtifacts": True}))["tasks"][0]
            assert sent_artifact == fetched["artifacts"][0]
            assert sent_status["timestamp"] == fetched["status"]["timestamp"]
            if not streaming:
                assert task == fetched
            frames, _ = await asyncio.to_thread(_post_sse, base + "/", {
                "jsonrpc": "2.0", "id": "replay", "method": "SubscribeToTask", "params": {"taskId": task_id}})
            assert frames[0]["artifactUpdate"]["artifact"] == sent_artifact
            assert frames[-1]["statusUpdate"]["status"]["timestamp"] == sent_status["timestamp"]
        finally:
            await adapter.disconnect()

    asyncio.run(run())

