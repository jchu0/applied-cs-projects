"""Tests for the real network/RPC transport (``paramserver.transport``).

These exercise actual TCP sockets on the loopback interface, proving the server
can be driven across a connection rather than only in-process.
"""

import numpy as np
import pytest

from paramserver.transport import (
    RemoteParameterServer,
    RpcClient,
    RpcError,
    RpcServer,
    serve_parameter_server,
)


async def test_rpc_echo_round_trip():
    server = RpcServer("127.0.0.1", 0)

    async def echo(params):
        return params.get("value")

    server.register("echo", echo)
    await server.start()
    client = await RpcClient("127.0.0.1", server.port).connect()
    try:
        assert await client.call("echo", value=[1, 2, 3]) == [1, 2, 3]
        # NumPy arrays survive the pickle transport intact.
        arr = await client.call("echo", value=np.arange(4, dtype=np.float32))
        assert np.allclose(arr, [0, 1, 2, 3])
    finally:
        await client.close()
        await server.stop()


async def test_rpc_unknown_method_raises():
    server = RpcServer("127.0.0.1", 0)
    await server.start()
    client = await RpcClient("127.0.0.1", server.port).connect()
    try:
        with pytest.raises(RpcError):
            await client.call("does_not_exist")
    finally:
        await client.close()
        await server.stop()


async def test_rpc_handler_error_propagates():
    server = RpcServer("127.0.0.1", 0)

    async def boom(_params):
        raise ValueError("kaboom")

    server.register("boom", boom)
    await server.start()
    client = await RpcClient("127.0.0.1", server.port).connect()
    try:
        with pytest.raises(RpcError, match="kaboom"):
            await client.call("boom")
    finally:
        await client.close()
        await server.stop()


async def test_parameter_server_over_rpc(parameter_server, small_params):
    """A real shard, served on a socket and driven from a remote client."""
    await parameter_server.initialize(small_params)
    rpc = await serve_parameter_server(parameter_server, "127.0.0.1", 0)
    client = await RemoteParameterServer("127.0.0.1", rpc.port).connect()
    try:
        # Pull initial values across the wire.
        pulled = await client.pull(["w1"], worker_id=1)
        assert "w1" in pulled
        value, version = pulled["w1"]
        assert np.allclose(value, small_params["w1"])

        # Push a gradient and confirm the server applied it.
        applied = await client.push(
            {"w1": np.ones_like(small_params["w1"])}, worker_id=1, clock=0
        )
        assert applied >= 1

        # Pull again: parameters changed and the version advanced — a real,
        # networked round-trip, not an in-process call.
        pulled2 = await client.pull(["w1"], worker_id=1)
        value2, version2 = pulled2["w1"]
        assert not np.allclose(value2, small_params["w1"])
        assert version2 > version
    finally:
        await client.close()
        await rpc.stop()
