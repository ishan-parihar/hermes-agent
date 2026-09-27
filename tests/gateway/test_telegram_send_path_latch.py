"""Regression contract: a fenced adapter must still delegate to a live replacement.

``disconnect()`` raises ``_send_path_degraded`` but ``_mark_disconnected()`` never clears
``_bot``. ``send()`` therefore skipped its whole recovery block (guarded by ``if not
self._bot``) and refused every send with ``send_path_degraded`` until the process was
restarted -- a 2h50m Telegram outage on 2026-09-27 that silently dropped 5 replies and
4 queued inbound messages. Delegating to a healthy replacement must win over the stale
fence, and the fence must still hold when no replacement exists.

Driven with ``asyncio.run`` rather than ``pytest-asyncio``: the runtime venv deliberately
omits test tooling (pyproject lists pytest-asyncio under the dev dependency-group, and
marks it absent from the runtime payload), so the async plugin is unavailable here.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _attach_replacement(adapter, live):
    runner = MagicMock()
    runner.adapters = {adapter.platform: live}
    adapter.gateway_runner = runner
    return live


def test_fenced_send_delegates_to_live_replacement():
    async def scenario():
        adapter = _make_adapter()
        adapter._bot = MagicMock()  # stale: disconnect() never cleared it
        adapter._send_path_degraded = True  # raised by disconnect()

        live = MagicMock()
        live._bot = MagicMock()
        live.send = AsyncMock(return_value=SendResult(success=True, message_id=7))
        _attach_replacement(adapter, live)

        return await adapter.send("123", "hello"), live

    result, live = asyncio.run(scenario())

    assert result.success is True
    assert result.message_id == 7
    live.send.assert_awaited_once_with("123", "hello", None, None)


def test_fenced_send_without_replacement_still_refuses():
    async def scenario():
        adapter = _make_adapter()
        adapter._bot = MagicMock()
        adapter._send_path_degraded = True
        return await adapter.send("123", "hello")

    result = asyncio.run(scenario())

    assert result.success is False
    assert result.error == "send_path_degraded"
    assert result.retryable is True


def test_unfenced_send_is_unchanged():
    async def scenario():
        adapter = _make_adapter()
        adapter._bot = MagicMock()
        adapter._send_path_degraded = False

        runner = MagicMock()
        runner.adapters = {}
        adapter.gateway_runner = runner

        sent = {}

        async def _fake_send_text_locked(chat_id, content, reply_to, metadata):
            sent["chat_id"] = chat_id
            sent["content"] = content
            return SendResult(success=True, message_id=11)

        adapter._send_text_locked = _fake_send_text_locked
        result = await adapter.send("123", "hello")
        return result, sent

    result, sent = asyncio.run(scenario())

    assert result.success is True
    assert result.message_id == 11
    assert sent == {"chat_id": "123", "content": "hello"}
