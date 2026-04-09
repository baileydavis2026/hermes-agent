"""
Sesame messaging platform adapter for Hermes Agent.

Connects Hermes to the Sesame messaging platform via WebSocket (real-time
inbound events) and REST API (outbound messages, file uploads).

Requires:
  - aiohttp and websockets packages installed
  - SESAME_API_KEY environment variable set
"""

import asyncio
import json
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp
import websockets
import websockets.exceptions

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_image_from_url,
    cache_audio_from_url,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 10_000

# Reconnect backoff settings
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_BACKOFF_MULTIPLIER = 2.0

# Heartbeat interval (server sends heartbeatIntervalMs on auth)
_DEFAULT_HEARTBEAT_INTERVAL = 30.0


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_api_url() -> str:
    return os.getenv("SESAME_API_URL", "https://api.sesame.space")


def _get_ws_url() -> str:
    return os.getenv("SESAME_WS_URL", "wss://ws.sesame.space")


def _get_api_key() -> Optional[str]:
    return os.getenv("SESAME_API_KEY")


def _get_allowed_users() -> set[str]:
    """Return the set of allowed principal IDs, or empty set for allow-all."""
    raw = os.getenv("SESAME_ALLOWED_USERS", "")
    if not raw.strip():
        return set()
    return {uid.strip() for uid in raw.split(",") if uid.strip()}


def _get_home_channel() -> Optional[str]:
    return os.getenv("SESAME_HOME_CHANNEL")


def check_sesame_requirements() -> bool:
    """Check that required dependencies and config are available."""
    if not _get_api_key():
        return False
    try:
        import aiohttp  # noqa: F401
        import websockets  # noqa: F401
        return True
    except ImportError as e:
        logger.error(
            "Missing dependency for Sesame adapter: %s — "
            "install with: pip install aiohttp websockets",
            e,
        )
        return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SesameAPIError(Exception):
    """Raised when a Sesame REST API call returns an error."""

    def __init__(self, status: int, message: str, body: dict):
        self.status = status
        self.message = message
        self.body = body
        super().__init__(f"Sesame API {status}: {message}")


class SesameAuthError(Exception):
    """Raised when WebSocket authentication fails."""
    pass


# ---------------------------------------------------------------------------
# Sesame REST + WebSocket Client
# ---------------------------------------------------------------------------

class SesameClient:
    """Lightweight async client for the Sesame messaging platform."""

    def __init__(
        self,
        api_url: str,
        ws_url: str,
        api_key: str,
    ):
        self.api_url = api_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self.api_key = api_key

        # Identity (populated on connect)
        self.principal_id: Optional[str] = None
        self.handle: Optional[str] = None
        self.display_name: Optional[str] = None
        self.workspace_id: Optional[str] = None

        # WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._running = False
        self._heartbeat_interval = _DEFAULT_HEARTBEAT_INTERVAL

        # Callbacks
        self._on_message: Optional[Callable[[dict], Coroutine]] = None
        self._on_connected: Optional[Callable[[], Coroutine]] = None

        # HTTP session (lazy)
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Make an authenticated REST request and return parsed JSON."""
        url = f"{self.api_url}{path}"
        session = self._get_session()
        async with session.request(
            method, url, json=json_body, params=params
        ) as resp:
            body = await resp.json()
            if resp.status >= 400:
                error = body.get("error", body.get("message", resp.reason))
                raise SesameAPIError(resp.status, error, body)
            return body

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    async def fetch_identity(self) -> dict:
        """GET /api/v1/auth/me — fetch the authenticated principal."""
        result = await self._request("GET", "/api/v1/auth/me")
        data = result.get("data", result)
        self.principal_id = data["id"]
        self.handle = data.get("handle")
        self.display_name = data.get("displayName")
        self.workspace_id = data.get("workspaceId")
        logger.info(
            "Sesame identity: %s (%s) principal=%s",
            self.handle,
            self.display_name,
            self.principal_id,
        )
        return data

    # ------------------------------------------------------------------
    # Messaging REST
    # ------------------------------------------------------------------

    async def send_message(
        self,
        channel_id: str,
        content: str,
        *,
        kind: str = "text",
        intent: str = "chat",
        thread_root_id: Optional[str] = None,
        attachment_ids: Optional[list[str]] = None,
        client_generated_id: Optional[str] = None,
    ) -> dict:
        """POST /api/v1/channels/:channelId/messages"""
        body: dict[str, Any] = {
            "content": content,
            "kind": kind,
            "intent": intent,
        }
        if thread_root_id:
            body["threadRootId"] = thread_root_id
        if attachment_ids:
            body["attachmentIds"] = attachment_ids
        if client_generated_id:
            body["clientGeneratedId"] = client_generated_id

        result = await self._request(
            "POST", f"/api/v1/channels/{channel_id}/messages", json_body=body
        )
        return result.get("data", result)

    async def edit_message(
        self,
        channel_id: str,
        message_id: str,
        content: str,
        *,
        streaming: bool = False,
    ) -> dict:
        """PATCH /api/v1/channels/:channelId/messages/:messageId"""
        body: dict[str, Any] = {"content": content}
        if streaming:
            body["streaming"] = True

        result = await self._request(
            "PATCH",
            f"/api/v1/channels/{channel_id}/messages/{message_id}",
            json_body=body,
        )
        return result.get("data", result)

    async def get_channel_info(self, channel_id: str) -> dict:
        """GET /api/v1/channels/:channelId"""
        result = await self._request("GET", f"/api/v1/channels/{channel_id}")
        return result.get("data", result)

    async def get_messages(
        self,
        channel_id: str,
        *,
        cursor: Optional[int] = None,
        limit: int = 50,
        direction: str = "before",
    ) -> dict:
        """GET /api/v1/channels/:channelId/messages (cursor-based)"""
        params: dict[str, Any] = {"limit": str(limit), "direction": direction}
        if cursor is not None:
            params["cursor"] = str(cursor)
        return await self._request(
            "GET", f"/api/v1/channels/{channel_id}/messages", params=params
        )

    # ------------------------------------------------------------------
    # Drive / file uploads
    # ------------------------------------------------------------------

    async def upload_file(
        self,
        file_path: str,
        file_name: str,
        content_type: str,
        size: int,
        *,
        channel_id: Optional[str] = None,
    ) -> str:
        """Upload a file via Drive presigned URL flow, return the file ID."""
        # Step 1: get presigned upload URL
        upload_meta = await self._request(
            "POST",
            "/api/v1/drive/files/upload-url",
            json_body={
                "fileName": file_name,
                "contentType": content_type,
                "size": size,
            },
        )
        data = upload_meta.get("data", upload_meta)
        upload_url = data["uploadUrl"]
        file_id = data["fileId"]
        s3_key = data["s3Key"]

        # Step 2: PUT file bytes to presigned URL
        session = self._get_session()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        async with session.put(
            upload_url,
            data=file_bytes,
            headers={"Content-Type": content_type},
        ) as resp:
            if resp.status >= 400:
                raise SesameAPIError(resp.status, "File upload failed", {})

        # Step 3: register the file
        register_body: dict[str, Any] = {
            "fileId": file_id,
            "s3Key": s3_key,
            "fileName": file_name,
            "contentType": content_type,
            "size": size,
        }
        if channel_id:
            register_body["channelId"] = channel_id
        await self._request(
            "POST", "/api/v1/drive/files", json_body=register_body
        )
        return file_id

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    def on_message(self, callback: Callable[[dict], Coroutine]):
        """Register a callback for incoming WebSocket events."""
        self._on_message = callback

    def on_connected(self, callback: Callable[[], Coroutine]):
        """Register a callback fired after successful WS auth."""
        self._on_connected = callback

    async def connect_ws(self):
        """Start the WebSocket listener with automatic reconnection."""
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())

    async def disconnect(self):
        """Close WebSocket and HTTP session."""
        self._running = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send_ws(self, frame: dict):
        """Send a JSON frame over the WebSocket."""
        if self._ws:
            await self._ws.send(json.dumps(frame))

    async def send_typing(self, channel_id: str):
        """Send a typing indicator via WebSocket."""
        await self.send_ws({"type": "typing", "channelId": channel_id})

    async def send_status(
        self,
        status: str,
        *,
        detail: Optional[str] = None,
        progress: Optional[int] = None,
    ):
        """Send a presence/status update via WebSocket."""
        frame: dict[str, Any] = {"type": "status", "status": status}
        if detail:
            frame["detail"] = detail
        if progress is not None:
            frame["progress"] = progress
        await self.send_ws(frame)

    async def send_runtime_meta(self, runtime: str, version: str):
        """Send runtime metadata frame (agent identification)."""
        await self.send_ws(
            {"type": "meta", "runtime": runtime, "version": version}
        )

    # ------------------------------------------------------------------
    # Internal: WebSocket event loop
    # ------------------------------------------------------------------

    async def _ws_loop(self):
        """Reconnecting WebSocket loop with exponential backoff."""
        backoff = _INITIAL_BACKOFF

        while self._running:
            try:
                await self._connect_and_listen()
                # Clean disconnect — reset backoff
                backoff = _INITIAL_BACKOFF
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                logger.warning(
                    "Sesame WebSocket disconnected: %s — reconnecting in %.1fs",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF)

    async def _connect_and_listen(self):
        """Single WebSocket connection lifecycle."""
        ws_uri = f"{self.ws_url}/v1/connect"
        logger.info("Connecting to Sesame WebSocket: %s", ws_uri)

        async with websockets.connect(
            ws_uri,
            ping_interval=None,  # we handle heartbeat ourselves
            max_size=10 * 1024 * 1024,  # 10 MB
            close_timeout=5,
        ) as ws:
            self._ws = ws

            # Authenticate
            await ws.send(json.dumps({"type": "auth", "apiKey": self.api_key}))

            # Wait for authenticated frame
            auth_response = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if auth_response.get("type") != "authenticated":
                error_msg = auth_response.get("error", "Authentication failed")
                raise SesameAuthError(error_msg)

            self._heartbeat_interval = (
                auth_response.get("heartbeatIntervalMs", 30000) / 1000.0
            )
            logger.info(
                "Sesame WebSocket authenticated (heartbeat=%.0fs)",
                self._heartbeat_interval,
            )

            # Send runtime metadata
            await self.send_runtime_meta("hermes", "1.0.0")

            if self._on_connected:
                await self._on_connected()

            # Start heartbeat and listener concurrently
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))
            try:
                await self._listen(ws)
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _heartbeat_loop(self, ws):
        """Send periodic pings to keep the connection alive."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await ws.send(json.dumps({"type": "ping"}))
            except Exception:
                break

    async def _listen(self, ws):
        """Read frames from the WebSocket and dispatch to handler."""
        async for raw in ws:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Sesame WS: invalid JSON frame: %s", raw[:200])
                continue

            frame_type = frame.get("type")

            # Skip internal frames
            if frame_type in ("pong", "authenticated", "delivery.ack"):
                continue

            if self._on_message:
                try:
                    await self._on_message(frame)
                except Exception:
                    logger.exception(
                        "Error handling Sesame WS frame type=%s", frame_type
                    )


# ---------------------------------------------------------------------------
# Sesame Platform Adapter
# ---------------------------------------------------------------------------

class SesameAdapter(BasePlatformAdapter):
    """Sesame messaging platform adapter for Hermes Agent."""

    platform = Platform.SESAME
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.SESAME)
        self.api_url = _get_api_url()
        self.ws_url = _get_ws_url()
        self.api_key = config.token or _get_api_key()
        if not self.api_key:
            raise ValueError(
                "Sesame API key is required. Set SESAME_API_KEY env var "
                "or pass token in PlatformConfig."
            )
        self.allowed_users = _get_allowed_users()
        self.home_channel = _get_home_channel()
        self.client = SesameClient(self.api_url, self.ws_url, self.api_key)

    # ----------------------------------------------------------
    # Connection lifecycle
    # ----------------------------------------------------------

    async def connect(self) -> bool:
        try:
            # 1. Fetch agent identity
            await self.client.fetch_identity()

            # 2. Register WS event handler
            self.client.on_message(self._on_ws_event)

            # 3. Connect WebSocket
            await self.client.connect_ws()

            self._mark_connected()
            logger.info(
                "Sesame adapter connected as %s (%s)",
                self.client.handle,
                self.client.principal_id,
            )
            return True

        except Exception as exc:
            logger.error("Sesame connection failed: %s", exc)
            self._set_fatal_error(
                "connect_failed",
                str(exc),
                retryable=True,
            )
            await self._notify_fatal_error()
            return False

    async def disconnect(self) -> None:
        self._mark_disconnected()
        await self.client.disconnect()
        logger.info("Sesame adapter disconnected")

    # ----------------------------------------------------------
    # Inbound: WebSocket event handling
    # ----------------------------------------------------------

    async def _on_ws_event(self, frame: dict):
        """Dispatch incoming WebSocket frames to Hermes."""
        frame_type = frame.get("type")

        if frame_type == "message":
            await self._handle_incoming_message(frame.get("message", {}))

    async def _handle_incoming_message(self, msg: dict):
        """Process an incoming message and dispatch to Hermes handler."""
        sender_id = msg.get("senderId")
        channel_id = msg.get("channelId")
        content = msg.get("plaintext") or msg.get("content") or ""
        kind = msg.get("kind", "text")
        message_id = msg.get("id")
        metadata = msg.get("metadata", {})
        sender_handle = metadata.get("senderHandle", "")
        sender_display = metadata.get("senderDisplayName", "")
        sender_kind = metadata.get("senderKind", "human")
        thread_root_id = msg.get("threadRootId")

        # Skip own messages
        if sender_id == self.client.principal_id:
            return

        # Skip non-text messages (system, call, etc.)
        if kind not in ("text",):
            return

        # Skip empty content
        if not content.strip():
            return

        # Authorization check
        if self.allowed_users and sender_id not in self.allowed_users:
            logger.debug(
                "Sesame: ignoring message from unauthorized user %s",
                sender_id,
            )
            return

        # Determine chat type
        chat_type = "channel"
        try:
            channel_info = await self.client.get_channel_info(channel_id)
            channel_name = channel_info.get("name", channel_id)
            ch_kind = channel_info.get("kind", "channel")
            if ch_kind == "dm":
                chat_type = "dm"
            elif ch_kind == "group":
                chat_type = "group"
        except Exception:
            channel_name = channel_id

        # Build session source
        source = SessionSource(
            platform=Platform.SESAME,
            chat_id=f"sesame:{channel_id}",
            chat_name=channel_name,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_display or sender_handle or sender_id,
            thread_id=thread_root_id,
        )

        # Extract media from attachments
        media_urls: list[str] = []
        media_types: list[str] = []
        attachments = metadata.get("attachments", [])
        for att in attachments:
            download_url = att.get("downloadUrl")
            content_type = att.get("contentType", "")
            if download_url:
                if content_type.startswith("image/"):
                    try:
                        ext = mimetypes.guess_extension(content_type) or ".jpg"
                        local_path = await cache_image_from_url(
                            download_url, ext
                        )
                        media_urls.append(local_path)
                        media_types.append("photo")
                    except Exception:
                        logger.warning(
                            "Failed to cache image attachment: %s",
                            att.get("fileName"),
                        )
                elif content_type.startswith("audio/"):
                    try:
                        ext = mimetypes.guess_extension(content_type) or ".ogg"
                        local_path = await cache_audio_from_url(
                            download_url, ext
                        )
                        media_urls.append(local_path)
                        media_types.append("audio")
                    except Exception:
                        logger.warning(
                            "Failed to cache audio attachment: %s",
                            att.get("fileName"),
                        )

        # Build message event
        msg_type = MessageType.TEXT
        if media_urls and media_types and media_types[0] == "photo":
            msg_type = MessageType.PHOTO
        elif media_urls and media_types and media_types[0] == "audio":
            msg_type = MessageType.AUDIO

        event = MessageEvent(
            text=content,
            message_type=msg_type,
            source=source,
            raw_message=msg,
            message_id=message_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=thread_root_id,
        )

        await self.handle_message(event)

    # ----------------------------------------------------------
    # Outbound: send messages
    # ----------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Send a text message to a Sesame channel."""
        channel_id = self._resolve_channel_id(chat_id)
        thread_root_id = None
        if metadata and metadata.get("thread_id"):
            thread_root_id = metadata["thread_id"]

        try:
            result = await self.client.send_message(
                channel_id,
                content,
                thread_root_id=thread_root_id,
            )
            return SendResult(
                success=True,
                message_id=result.get("id"),
                raw_response=result,
            )
        except SesameAPIError as exc:
            retryable = exc.status >= 500 or exc.status == 429
            return SendResult(
                success=False,
                error=str(exc),
                retryable=retryable,
            )
        except Exception as exc:
            return SendResult(
                success=False,
                error=str(exc),
                retryable=True,
            )

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ):
        """Edit a previously sent message (used for streaming edits)."""
        channel_id = self._resolve_channel_id(chat_id)
        try:
            result = await self.client.edit_message(
                channel_id,
                message_id,
                content,
                streaming=True,
            )
            return SendResult(
                success=True,
                message_id=message_id,
                raw_response=result,
            )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator via WebSocket."""
        channel_id = self._resolve_channel_id(chat_id)
        try:
            await self.client.send_typing(channel_id)
        except Exception:
            pass  # typing indicators are best-effort

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Upload image via Drive and send as attachment."""
        channel_id = self._resolve_channel_id(chat_id)
        try:
            file_id = await self._upload_from_path_or_url(
                image_url, channel_id
            )
            content = caption or ""
            result = await self.client.send_message(
                channel_id,
                content,
                attachment_ids=[file_id],
            )
            return SendResult(
                success=True,
                message_id=result.get("id"),
                raw_response=result,
            )
        except Exception as exc:
            logger.warning("Sesame send_image failed: %s", exc)
            # Fall back to sending URL as text
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(chat_id, text, reply_to, metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ):
        """Upload file via Drive and send as attachment."""
        channel_id = self._resolve_channel_id(chat_id)
        try:
            path = Path(file_path)
            name = file_name or path.name
            content_type = (
                mimetypes.guess_type(name)[0] or "application/octet-stream"
            )
            size = path.stat().st_size
            fid = await self.client.upload_file(
                file_path, name, content_type, size, channel_id=channel_id
            )
            content = caption or ""
            result = await self.client.send_message(
                channel_id, content, attachment_ids=[fid]
            )
            return SendResult(
                success=True,
                message_id=result.get("id"),
                raw_response=result,
            )
        except Exception as exc:
            logger.warning("Sesame send_document failed: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return channel info dict."""
        channel_id = self._resolve_channel_id(chat_id)
        try:
            info = await self.client.get_channel_info(channel_id)
            ch_kind = info.get("kind", "channel")
            return {
                "name": info.get("name", channel_id),
                "type": ch_kind if ch_kind in ("dm", "group") else "channel",
                "chat_id": chat_id,
            }
        except Exception:
            return {"name": chat_id, "type": "channel", "chat_id": chat_id}

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _resolve_channel_id(self, chat_id: str) -> str:
        """Strip the ``sesame:`` prefix added during session keying."""
        if chat_id.startswith("sesame:"):
            return chat_id[7:]
        return chat_id

    async def _upload_from_path_or_url(
        self, path_or_url: str, channel_id: str
    ) -> str:
        """Upload a local file or download-then-upload a URL. Returns file ID."""
        p = Path(path_or_url)
        if p.exists():
            name = p.name
            content_type = (
                mimetypes.guess_type(name)[0] or "application/octet-stream"
            )
            size = p.stat().st_size
            return await self.client.upload_file(
                str(p), name, content_type, size, channel_id=channel_id
            )
        else:
            # It's a URL — download to temp, then upload
            import tempfile

            async with aiohttp.ClientSession() as sess:
                async with sess.get(path_or_url) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
                    content_type = resp.content_type or "application/octet-stream"

            ext = mimetypes.guess_extension(content_type) or ""
            name = f"upload_{uuid.uuid4().hex[:8]}{ext}"

            tmp = Path(tempfile.mkdtemp()) / name
            tmp.write_bytes(data)
            try:
                return await self.client.upload_file(
                    str(tmp),
                    name,
                    content_type,
                    len(data),
                    channel_id=channel_id,
                )
            finally:
                tmp.unlink(missing_ok=True)
