"""Concurrent immutable dispatch leases; settings changes wait for actual callers."""

import asyncio
from contextlib import asynccontextmanager


class BindingBarrier:
    def __init__(self):
        self.condition = asyncio.Condition()
        self.readers = 0
        self.writers = 0
        self.writing = False

    def locked(self):
        return self.writing or self.writers > 0

    @asynccontextmanager
    async def read(self):
        async with self.condition:
            await self.condition.wait_for(lambda: not self.writing and not self.writers)
            self.readers += 1
        try:
            yield
        finally:
            async with self.condition:
                self.readers -= 1
                self.condition.notify_all()

    async def __aenter__(self):
        async with self.condition:
            self.writers += 1
            try:
                await self.condition.wait_for(
                    lambda: not self.writing and not self.readers
                )
                self.writing = True
            finally:
                self.writers -= 1
                self.condition.notify_all()
        return self

    async def __aexit__(self, *_):
        async with self.condition:
            self.writing = False
            self.condition.notify_all()
