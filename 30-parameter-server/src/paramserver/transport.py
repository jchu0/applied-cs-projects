"""Real network/RPC transport for the parameter server.

The core server and workers otherwise communicate through in-process asyncio
primitives only — they cannot span processes or machines. This module adds a
genuine async RPC layer over TCP using nothing but the standard library
(`asyncio`), so a :class:`~paramserver.server.parameter_server.ParameterServer`
shard can be served on a socket and a worker can ``pull``/``push`` against it
over the network.

Wire protocol: each message is a 4-byte big-endian length prefix followed by a
pickled ``dict``. Pickle is used because gradients and parameters are NumPy
arrays; **only use this on a trusted internal network** — unpickling executes
arbitrary code, so it must never be exposed to untrusted clients.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import struct
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)

_LEN = struct.Struct("!I")  # 4-byte big-endian unsigned length prefix
_MAX_MSG_BYTES = 256 * 1024 * 1024  # 256 MiB guard against absurd frames

#: An RPC handler: receives the request ``params`` dict, returns a result.
Handler = Callable[[Dict[str, Any]], Awaitable[Any]]


class RpcError(Exception):
    """Raised on the client when the server reports a failed call."""


async def _read_msg(reader: asyncio.StreamReader) -> Any:
    header = await reader.readexactly(_LEN.size)
    (length,) = _LEN.unpack(header)
    if length > _MAX_MSG_BYTES:
        raise RpcError(f"message of {length} bytes exceeds limit {_MAX_MSG_BYTES}")
    data = await reader.readexactly(length)
    return pickle.loads(data)


async def _write_msg(writer: asyncio.StreamWriter, obj: Any) -> None:
    data = pickle.dumps(obj)
    writer.write(_LEN.pack(len(data)) + data)
    await writer.drain()


class RpcServer:
    """A minimal async RPC server: register named handlers, then ``start``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self._handlers: Dict[str, Handler] = {}
        self._server: asyncio.AbstractServer | None = None

    def register(self, method: str, handler: Handler) -> None:
        """Register an async handler for ``method``."""
        self._handlers[method] = handler

    async def start(self) -> "RpcServer":
        """Bind and start serving. When ``port == 0`` an ephemeral port is chosen
        and stored back on :attr:`port`."""
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        logger.info("RPC server listening on %s:%d", self.host, self.port)
        return self

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            while True:
                try:
                    req = await _read_msg(reader)
                except asyncio.IncompleteReadError:
                    break  # client closed the connection cleanly
                method = req.get("method")
                params = req.get("params", {})
                handler = self._handlers.get(method)
                if handler is None:
                    resp = {"ok": False, "error": f"unknown method: {method}"}
                else:
                    try:
                        result = await handler(params)
                        resp = {"ok": True, "result": result}
                    except Exception as exc:  # surface handler errors to the caller
                        logger.exception("RPC handler %r failed", method)
                        resp = {"ok": False, "error": str(exc)}
                await _write_msg(writer, resp)
        except (ConnectionError, RpcError) as exc:
            logger.warning("RPC connection from %s dropped: %s", peer, exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover - best-effort close
                pass

    async def stop(self) -> None:
        """Stop accepting connections and wait for shutdown."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


class RpcClient:
    """A connection to an :class:`RpcServer`. Calls are serialized per connection."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> "RpcClient":
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        return self

    async def call(self, method: str, **params: Any) -> Any:
        """Invoke ``method`` on the server and return its result, or raise
        :class:`RpcError` if the server reports failure."""
        if self._writer is None or self._reader is None:
            raise RpcError("client is not connected; call connect() first")
        async with self._lock:  # at most one in-flight request per connection
            await _write_msg(self._writer, {"method": method, "params": params})
            resp = await _read_msg(self._reader)
        if not resp.get("ok"):
            raise RpcError(resp.get("error", "unknown RPC error"))
        return resp.get("result")

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:  # pragma: no cover - best-effort close
                pass
            self._writer = None
            self._reader = None


async def serve_parameter_server(
    server: Any, host: str = "127.0.0.1", port: int = 0
) -> RpcServer:
    """Expose a :class:`ParameterServer` shard over RPC.

    Registers ``pull`` and ``push`` methods that delegate to the shard's async
    API, then starts listening. Returns the running :class:`RpcServer` (use
    ``rpc.port`` for the bound port and ``await rpc.stop()`` to shut down).
    """
    rpc = RpcServer(host, port)

    async def _pull(params: Dict[str, Any]) -> Any:
        return await server.pull(
            params["param_names"],
            params.get("worker_id", -1),
            params.get("include_versions", True),
        )

    async def _push(params: Dict[str, Any]) -> Any:
        return await server.push(
            params["gradients"],
            params.get("worker_id", -1),
            params.get("clock", 0),
        )

    rpc.register("pull", _pull)
    rpc.register("push", _push)
    await rpc.start()
    return rpc


class RemoteParameterServer:
    """Client-side proxy mirroring the shard API over the network.

    Drop-in for the in-process ``ParameterServer`` from a worker's perspective:
    ``pull`` and ``push`` behave the same, but travel over a real socket.
    """

    def __init__(self, host: str, port: int):
        self._client = RpcClient(host, port)

    async def connect(self) -> "RemoteParameterServer":
        await self._client.connect()
        return self

    async def pull(self, param_names, worker_id: int = -1, include_versions: bool = True):
        return await self._client.call(
            "pull",
            param_names=param_names,
            worker_id=worker_id,
            include_versions=include_versions,
        )

    async def push(self, gradients, worker_id: int = -1, clock: int = 0) -> int:
        return await self._client.call(
            "push", gradients=gradients, worker_id=worker_id, clock=clock
        )

    async def close(self) -> None:
        await self._client.close()
