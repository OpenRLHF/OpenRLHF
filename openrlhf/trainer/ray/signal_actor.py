import asyncio
from enum import Enum, auto

import ray


class SignalState(Enum):
    IDLE = auto()
    GENERATOR = auto()
    UPDATE_WEIGHT = auto()


@ray.remote(num_cpus=0)
class SignalActor:
    """Cross-actor signal for mutual exclusion between generation and weight update.

    Replaces asyncio.Event with asyncio.Condition + state enum to guarantee
    that rollout generation and weight broadcast never overlap, preventing
    model weight dirty writes and abnormal sample generation.

    State transitions:
        IDLE -> GENERATOR   (acquired by generator)
        IDLE -> UPDATE_WEIGHT (acquired by weight updater)
        GENERATOR -> IDLE   (released by generator)
        UPDATE_WEIGHT -> IDLE (released by weight updater)
    """

    def __init__(self):
        self._cond = asyncio.Condition()
        self._state = SignalState.IDLE

    async def set_generating(self):
        """Acquire the signal for trajectory generation.

        Blocks until the signal is IDLE, then transitions to GENERATOR.
        Call set_idle() to release.
        """
        async with self._cond:
            while self._state != SignalState.IDLE:
                await self._cond.wait()
            self._state = SignalState.GENERATOR

    async def set_update_weights(self):
        """Acquire the signal for weight update.

        Blocks until the signal is IDLE, then transitions to UPDATE_WEIGHT.
        Call set_idle() to release.
        """
        async with self._cond:
            while self._state != SignalState.IDLE:
                await self._cond.wait()
            self._state = SignalState.UPDATE_WEIGHT

    async def set_idle(self):
        """Release the signal, returning to IDLE and waking all waiters."""
        async with self._cond:
            self._state = SignalState.IDLE
            self._cond.notify_all()

    async def get_state(self):
        """Return the current state name (for diagnostics)."""
        async with self._cond:
            return self._state.name
