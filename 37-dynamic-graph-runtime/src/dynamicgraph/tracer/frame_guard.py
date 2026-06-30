"""Frame guards for tracing."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional


class GuardCondition(Enum):
    """Types of guard conditions."""
    TYPE_MATCH = auto()
    SHAPE_MATCH = auto()
    VALUE_MATCH = auto()
    ATTRIBUTE_MATCH = auto()


@dataclass
class GuardFailure:
    """Represents a guard failure."""
    condition: GuardCondition
    expected: Any
    actual: Any
    message: str = ""

    def __str__(self) -> str:
        return f"Guard failed: {self.condition.name} - expected {self.expected}, got {self.actual}"


class FrameGuard:
    """
    Guard for checking frame validity during tracing.

    Guards ensure that traced code paths remain valid across invocations.
    """

    def __init__(self):
        self.conditions: list[tuple[GuardCondition, Callable[[], bool]]] = []
        self.failures: list[GuardFailure] = []

    def add_condition(
        self,
        condition: GuardCondition,
        check: Callable[[], bool],
        expected: Any = None,
        actual_fn: Callable[[], Any] = None,
    ) -> None:
        """Add a guard condition."""
        self.conditions.append((condition, check))

    def check(self) -> bool:
        """Check all guard conditions."""
        self.failures = []
        for condition, check in self.conditions:
            if not check():
                self.failures.append(
                    GuardFailure(condition=condition, expected=None, actual=None)
                )
        return len(self.failures) == 0

    def get_failures(self) -> list[GuardFailure]:
        """Get list of guard failures."""
        return self.failures

    def reset(self) -> None:
        """Reset guard state."""
        self.conditions = []
        self.failures = []
