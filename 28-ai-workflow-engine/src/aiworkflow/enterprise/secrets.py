"""Secret management for workflows.

Workflow definitions should never embed credentials directly. Instead they
reference secrets as ``${secret:NAME}`` and a :class:`SecretResolver` injects
the real values at run time from a pluggable provider (environment, in-memory,
or a vault adapter). A masking helper keeps resolved secrets out of logs.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

# Matches ${secret:NAME} where NAME is alphanumeric / underscore / dash / dot.
_SECRET_REF = re.compile(r"\$\{secret:([A-Za-z0-9_.\-]+)\}")


class SecretProvider(ABC):
    """Source of secret values, keyed by name."""

    @abstractmethod
    def get(self, name: str) -> Optional[str]:
        """Return the secret value for ``name``, or None if unknown."""


class InMemorySecretProvider(SecretProvider):
    """Secret provider backed by a dict (tests, defaults)."""

    def __init__(self, secrets: dict[str, str] = None):
        self._secrets = dict(secrets or {})

    def set(self, name: str, value: str) -> None:
        self._secrets[name] = value

    def get(self, name: str) -> Optional[str]:
        return self._secrets.get(name)


class EnvSecretProvider(SecretProvider):
    """Secret provider backed by environment variables.

    Args:
        prefix: optional prefix; a reference ``${secret:OPENAI_KEY}`` with
            prefix ``WF_`` reads the ``WF_OPENAI_KEY`` environment variable.
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def get(self, name: str) -> Optional[str]:
        return os.environ.get(f"{self.prefix}{name}")


class ChainedSecretProvider(SecretProvider):
    """Tries multiple providers in order, returning the first hit."""

    def __init__(self, *providers: SecretProvider):
        self.providers = list(providers)

    def get(self, name: str) -> Optional[str]:
        for provider in self.providers:
            value = provider.get(name)
            if value is not None:
                return value
        return None


class SecretResolver:
    """Resolves ``${secret:NAME}`` references against a provider."""

    def __init__(self, provider: SecretProvider, strict: bool = True):
        """Initialize.

        Args:
            provider: where to fetch secret values.
            strict: if True, an unknown secret raises ``KeyError``; otherwise
                the reference is left untouched.
        """
        self.provider = provider
        self.strict = strict
        self._resolved: set[str] = set()  # values seen, for masking

    def resolve_string(self, text: str) -> str:
        """Replace every ``${secret:NAME}`` in ``text`` with its value."""

        def _sub(match: re.Match) -> str:
            name = match.group(1)
            value = self.provider.get(name)
            if value is None:
                if self.strict:
                    raise KeyError(f"Unknown secret: {name}")
                return match.group(0)
            self._resolved.add(value)
            return value

        return _SECRET_REF.sub(_sub, text)

    def resolve(self, obj: Any) -> Any:
        """Recursively resolve secret references in dicts/lists/strings."""
        if isinstance(obj, str):
            return self.resolve_string(obj)
        if isinstance(obj, dict):
            return {k: self.resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.resolve(v) for v in obj]
        return obj

    def mask(self, text: str, placeholder: str = "***") -> str:
        """Redact any resolved secret values from ``text`` (for safe logging)."""
        for value in self._resolved:
            if value:
                text = text.replace(value, placeholder)
        return text

    @staticmethod
    def references(obj: Any) -> set[str]:
        """Return the set of secret names referenced anywhere in ``obj``."""
        names: set[str] = set()
        if isinstance(obj, str):
            names.update(_SECRET_REF.findall(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                names |= SecretResolver.references(v)
        elif isinstance(obj, list):
            for v in obj:
                names |= SecretResolver.references(v)
        return names
