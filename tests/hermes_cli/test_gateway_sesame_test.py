"""Tests for `hermes gateway test sesame`."""

from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway


class _FakeSesameClient:
    def __init__(self, api_url, ws_url, api_key):
        self.api_url = api_url
        self.ws_url = ws_url
        self.api_key = api_key
        self.sent = []
        self.disconnected = False

    async def fetch_identity(self):
        return {
            "id": "agent-123",
            "handle": "hermes-bot",
            "displayName": "Hermes Bot",
            "workspaceId": "workspace-1",
        }

    async def connect_ws(self):
        return None

    async def disconnect(self):
        self.disconnected = True

    async def send_message(self, channel_id, content, **kwargs):
        self.sent.append((channel_id, content, kwargs))
        return {"id": "msg-1"}


def test_sesame_gateway_test_reports_missing_api_key(monkeypatch, capsys):
    monkeypatch.delenv("SESAME_API_KEY", raising=False)

    code = gateway.run_sesame_gateway_test(SimpleNamespace(send_test_message=False, channel=None))

    out = capsys.readouterr().out
    assert code == 1
    assert "SESAME_API_KEY" in out
    assert "hermes gateway setup" in out


def test_sesame_gateway_test_validates_identity_and_ws(monkeypatch, capsys):
    monkeypatch.setenv("SESAME_API_KEY", "sk_test_fake")
    monkeypatch.setenv("SESAME_API_URL", "https://api.example.test")
    monkeypatch.setenv("SESAME_WS_URL", "wss://ws.example.test")
    monkeypatch.setattr(gateway, "_sesame_requirements_ok", lambda: True)
    monkeypatch.setattr(gateway, "_SesameTestClient", _FakeSesameClient)

    code = gateway.run_sesame_gateway_test(SimpleNamespace(send_test_message=False, channel=None))

    out = capsys.readouterr().out
    assert code == 0
    assert "Authenticated as @hermes-bot" in out
    assert "WebSocket auth succeeded" in out
    assert "SESAME_HOME_CHANNEL" in out


def test_sesame_gateway_test_can_send_message_to_explicit_channel(monkeypatch, capsys):
    clients = []

    class RecordingClient(_FakeSesameClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setenv("SESAME_API_KEY", "sk_test_fake")
    monkeypatch.setattr(gateway, "_sesame_requirements_ok", lambda: True)
    monkeypatch.setattr(gateway, "_SesameTestClient", RecordingClient)

    code = gateway.run_sesame_gateway_test(
        SimpleNamespace(send_test_message=True, channel="sesame:channel-123")
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "Sent test message" in out
    assert clients[0].sent[0][0] == "channel-123"
    assert clients[0].sent[0][2]["intent"] == "notification"
