"""Wire-boundary regression test for the duplicate ``reasoning_content`` 400 (t_79a61648).

A crash body from 2026-10-13 carried BOTH the trajectory-only ``reasoning`` key and
``reasoning_content`` on an assistant tool-call row; serde backends that alias
``reasoning`` -> ``reasoning_content`` (NVIDIA integrate API) deserialize both into
the same target field and reject with HTTP 400 "duplicate field reasoning_content".
turn_context pops the key at build time, but a long-lived process (or any future
build-path regression) can still leak it, so the chat-completions transport strips
it at the wire boundary too. These tests pin that boundary.
"""

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport, _STRIP_MSG_KEYS


def _tool_call_row() -> dict:
    return {
        "role": "assistant",
        "content": "",
        "reasoning": "Let me start by running the preflight and reading state.",
        "reasoning_content": "Let me start by running the preflight and reading state.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{\"command\": \"ls\"}"},
            }
        ],
    }


def test_strip_keyset_contains_trajectory_only_reasoning_keys():
    assert "reasoning" in _STRIP_MSG_KEYS
    assert "finish_reason" in _STRIP_MSG_KEYS


def test_convert_messages_strips_leaked_reasoning_key():
    transport = ChatCompletionsTransport()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        _tool_call_row(),
        {"role": "tool", "content": "out", "tool_call_id": "call_1"},
    ]
    out = transport.convert_messages(
        messages, model="free-stack", base_url="http://127.0.0.1:20129/v1"
    )
    for msg in out:
        assert "reasoning" not in msg
        assert "finish_reason" not in msg
    assistant = next(m for m in out if m.get("role") == "assistant")
    # The provider-facing field survives; only the trajectory-only alias is stripped.
    assert assistant["reasoning_content"].startswith("Let me start")
    assert assistant["tool_calls"][0]["function"]["name"] == "terminal"


def test_convert_messages_noop_when_clean():
    transport = ChatCompletionsTransport()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "done"},
    ]
    out = transport.convert_messages(
        messages, model="free-stack", base_url="http://127.0.0.1:20129/v1"
    )
    # Clean input must not be copied: the same list object comes back.
    assert out is messages


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
