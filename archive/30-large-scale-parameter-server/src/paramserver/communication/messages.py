"""Message handling for parameter server communication."""

import asyncio
import time
from typing import Any
import logging
import numpy as np

from ..schemas import (
    PullRequest,
    PushRequest,
    Parameter,
    Gradient,
    generate_id,
)

logger = logging.getLogger(__name__)


class MessageHandler:
    """Handles messages between workers and servers."""

    def __init__(self, server):
        """Initialize message handler.

        Args:
            server: Parameter server instance
        """
        self.server = server
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._response_futures: dict[str, asyncio.Future] = {}
        self._running = False

    async def start(self):
        """Start message processing."""
        self._running = True
        asyncio.create_task(self._process_loop())

    async def stop(self):
        """Stop message processing."""
        self._running = False

    async def _process_loop(self):
        """Process message queue."""
        while self._running:
            try:
                msg_type, request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=1.0
                )
                await self._handle_message(msg_type, request)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")

    async def _handle_message(self, msg_type: str, request: Any):
        """Handle a single message.

        Args:
            msg_type: Message type
            request: Request data
        """
        if msg_type == "pull":
            response = await self._handle_pull(request)
        elif msg_type == "push":
            response = await self._handle_push(request)
        elif msg_type == "barrier":
            response = await self._handle_barrier(request)
        else:
            response = {"error": f"Unknown message type: {msg_type}"}

        # Set response
        if request.request_id in self._response_futures:
            self._response_futures[request.request_id].set_result(response)

    async def _handle_pull(self, request: PullRequest) -> dict[str, Parameter]:
        """Handle pull request.

        Args:
            request: Pull request

        Returns:
            Parameters
        """
        return await self.server.pull(
            request.worker_id,
            request.parameter_names
        )

    async def _handle_push(self, request: PushRequest) -> dict[str, Any]:
        """Handle push request.

        Args:
            request: Push request

        Returns:
            Push result
        """
        return await self.server.push(
            request.worker_id,
            request.gradients,
            request.iteration
        )

    async def _handle_barrier(self, request: dict) -> dict[str, Any]:
        """Handle barrier request.

        Args:
            request: Barrier request

        Returns:
            Barrier status
        """
        return await self.server.barrier(
            request["worker_id"],
            request["iteration"]
        )

    async def send_request(
        self,
        msg_type: str,
        request: Any,
        timeout: float = 30.0
    ) -> Any:
        """Send request and wait for response.

        Args:
            msg_type: Message type
            request: Request data
            timeout: Timeout in seconds

        Returns:
            Response data
        """
        # Create future for response
        future = asyncio.Future()
        self._response_futures[request.request_id] = future

        # Add to queue
        await self._request_queue.put((msg_type, request))

        try:
            response = await asyncio.wait_for(future, timeout)
            return response
        finally:
            self._response_futures.pop(request.request_id, None)


class WorkerClient:
    """Client for workers to communicate with parameter server."""

    def __init__(self, worker_id: str, server_host: str = "localhost", server_port: int = 5000):
        """Initialize worker client.

        Args:
            worker_id: Worker identifier
            server_host: Server host
            server_port: Server port
        """
        self.worker_id = worker_id
        self.server_host = server_host
        self.server_port = server_port
        self._server = None

    def set_server(self, server):
        """Set server reference (for in-process testing).

        Args:
            server: Server instance
        """
        self._server = server

    async def pull(
        self,
        parameter_names: list[str],
        min_version: int = 0
    ) -> dict[str, Parameter]:
        """Pull parameters from server.

        Args:
            parameter_names: Parameters to pull
            min_version: Minimum version required

        Returns:
            Parameters
        """
        if self._server:
            return await self._server.pull(self.worker_id, parameter_names)

        # Would use RPC in production
        request = PullRequest(
            request_id=generate_id(),
            worker_id=self.worker_id,
            parameter_names=parameter_names,
            min_version=min_version
        )
        return {"mock": True}

    async def push(
        self,
        gradients: list[Gradient],
        iteration: int
    ) -> dict[str, Any]:
        """Push gradients to server.

        Args:
            gradients: Gradients to push
            iteration: Current iteration

        Returns:
            Push result
        """
        if self._server:
            return await self._server.push(
                self.worker_id,
                gradients,
                iteration
            )

        # Would use RPC in production
        return {"mock": True}

    async def barrier(self, iteration: int) -> bool:
        """Wait at synchronization barrier.

        Args:
            iteration: Iteration number

        Returns:
            True when barrier is complete
        """
        if self._server:
            return await self._server.barrier(self.worker_id, iteration)

        return True

    async def heartbeat(self) -> bool:
        """Send heartbeat to server.

        Returns:
            True if successful
        """
        if self._server:
            return await self._server.worker_heartbeat(self.worker_id)

        return True
