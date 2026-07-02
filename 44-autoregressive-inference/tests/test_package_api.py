"""Tests guarding the documented package-root public API.

These tests assert that the full public surface documented in the README and
docs/BLUEPRINT.md is exported from ``autoregressive_inference`` (the package
root) and that ``__all__`` stays consistent with the importable names. They
exist to catch export drift, e.g. a documented Protocol that is only reachable
from a submodule but missing from the top-level package.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import autoregressive_inference as ari


# The full public API as documented in README.md and docs/BLUEPRINT.md.
DOCUMENTED_API = [
    # Requests
    "RequestStatus",
    "RequestPriority",
    "SamplingParams",
    "InferenceRequest",
    "RequestManager",
    # KV cache
    "KVCacheConfig",
    "KVCacheBlock",
    "PagedKVCacheManager",
    "SlidingWindowCache",
    # Batching
    "ContinuousBatcher",
    "BatchedInputs",
    "SchedulingPolicy",
    "FIFOPolicy",
    "PriorityPolicy",
    "ShortestJobFirstPolicy",
    # Sampling
    "TokenSampler",
    # Scheduler
    "InferenceScheduler",
    "TransformerModel",
    # Speculative
    "SpeculativeDecoder",
    "TreeSpeculativeDecoder",
    "SpeculativeStats",
    "DraftModel",
    "TargetModel",
]


class TestPublicAPIExports:
    def test_all_documented_names_importable_from_root(self):
        """Every documented public name resolves on the package root."""
        missing = [name for name in DOCUMENTED_API if not hasattr(ari, name)]
        assert not missing, f"documented API missing from package root: {missing}"

    def test_all_matches_documented_api(self):
        """``__all__`` covers exactly the documented public surface."""
        assert set(ari.__all__) == set(DOCUMENTED_API)

    def test_every_all_entry_resolves(self):
        """No dangling names in ``__all__``."""
        unresolved = [name for name in ari.__all__ if not hasattr(ari, name)]
        assert not unresolved, f"__all__ names not resolvable: {unresolved}"

    def test_model_protocols_all_exported(self):
        """All three documented model-agnostic Protocols are exported."""
        # README lists TransformerModel, DraftModel, TargetModel together as the
        # model-agnostic interfaces; all must be reachable from the root.
        from autoregressive_inference import (
            TransformerModel,
            DraftModel,
            TargetModel,
        )

        assert TransformerModel is not None
        assert DraftModel is not None
        assert TargetModel is not None

    def test_version_exposed(self):
        assert isinstance(ari.__version__, str)
        assert ari.__version__
