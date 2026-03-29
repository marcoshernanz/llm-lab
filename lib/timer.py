"""Measure named durations in small, inspectable experiment scripts."""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


class Timer:
    """Small named timer registry for experiments.

    Example:
        timer = Timer()
        timer.start("total")
        timer.start("training")
        training_seconds = timer.stop("training")
        total_seconds = timer.stop("total")
    """

    def __init__(self) -> None:
        """Initialize empty running and completed timer registries."""
        self._starts: dict[str, float] = {}
        self._completed: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start a named timer."""
        if name in self._starts:
            raise ValueError(f"Timer {name!r} is already running.")
        self._starts[name] = perf_counter()

    def stop(self, name: str) -> float:
        """Stop a named timer and return its elapsed seconds."""
        if name not in self._starts:
            raise ValueError(f"Timer {name!r} was not started.")

        elapsed = perf_counter() - self._starts.pop(name)
        self._completed[name] = elapsed
        return elapsed

    def elapsed(self, name: str) -> float:
        """Return the current or completed duration for a timer."""
        if name in self._starts:
            return perf_counter() - self._starts[name]
        if name in self._completed:
            return self._completed[name]
        raise ValueError(f"Timer {name!r} has no recorded duration.")

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure a block with a named context manager."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
