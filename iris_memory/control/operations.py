"""Original-operation dispatch boundary, with no production business descriptors.

A future adapter must supply a finite reviewed descriptor and an exact transport.
This legacy descriptor registry remains empty. S2 uses dedicated finite ingress
and management adapters; tests here inject synthetic descriptors.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from ..core_client.protocol import Reply
from ..errors import ControlError
from .store import TERMINAL, Store


@dataclass(frozen=True)
class OperationDescriptor:
    kind: str
    validate: Callable[[dict], None]
    submit: Callable[[str, dict], Awaitable[Reply]]
    confirm: Callable[[str, dict], Awaitable[Reply]]


class OriginalOperations:
    def __init__(self, store: Store, descriptors: tuple[OperationDescriptor, ...] = ()):
        self.store = store
        self.descriptors = {d.kind: d for d in descriptors}
        self.busy = set()

    async def begin(self, binding, instance_id, kind, key, payload):
        descriptor = self.descriptors.get(kind)
        if descriptor is None:
            raise ControlError("OPERATION_NOT_IMPLEMENTED", 409)
        descriptor.validate(payload)
        operation, created = await self.store.create_operation(
            binding, instance_id, kind, key, payload
        )
        if not created:
            return (
                operation  # Never replay a submission, including after a lost response.
            )
        return await self._dispatch(operation, descriptor.submit)

    async def confirm(self, op_id):
        operation = await self.store.operation(op_id)
        descriptor = self.descriptors.get(operation["kind"])
        if descriptor is None:
            raise ControlError("OPERATION_NOT_IMPLEMENTED", 409)
        if operation["state"] in TERMINAL and not operation["cleanup_pending"]:
            return operation
        return await self._dispatch(operation, descriptor.confirm)

    async def _dispatch(self, operation, call):
        op_id = operation["id"]
        if op_id in self.busy:
            raise ControlError("OPERATION_BUSY", 409)
        if len(self.busy) >= 4:
            raise ControlError("OPERATION_ADMISSION_FULL", 429)
        self.busy.add(op_id)
        try:
            # Recheck at the last local boundary; the application serializes connection changes.
            await self.store.operation(op_id)
            async with asyncio.timeout(10):
                reply = await call(
                    operation["original_key"], json.loads(operation["original_input"])
                )
            await self.store.finish_operation(op_id, reply)
            return await self.store.operation(op_id, check_binding=False)
        finally:
            # Exceptions/cancellation preserve the already durable UNKNOWN original input.
            self.busy.remove(op_id)
