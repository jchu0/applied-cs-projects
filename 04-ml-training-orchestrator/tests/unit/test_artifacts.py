"""Tests for the artifact store."""

import pytest

from ml_orchestrator.experiment.artifacts import ArtifactStore


@pytest.mark.asyncio
async def test_save_load_get(tmp_path):
    store = ArtifactStore(base_path=str(tmp_path))
    art = await store.save_artifact("run-1", "model.bin", b"weights", artifact_type="model")
    assert art.run_id == "run-1"
    assert art.name == "model.bin"

    loaded = await store.load_artifact(art.id)
    assert loaded is not None
    meta, data = loaded
    assert data == b"weights"
    assert (await store.get_artifact(art.id)).id == art.id


@pytest.mark.asyncio
async def test_list_delete_size_stats(tmp_path):
    store = ArtifactStore(base_path=str(tmp_path))
    a1 = await store.save_artifact("run-1", "a", b"12345")
    await store.save_artifact("run-1", "b", b"67")

    assert len(await store.list_artifacts(run_id="run-1")) == 2
    assert await store.get_total_size("run-1") > 0

    assert await store.delete_artifact(a1.id) is True
    assert len(await store.list_artifacts(run_id="run-1")) == 1

    removed = await store.delete_run_artifacts("run-1")
    assert removed == 1
    assert isinstance(await store.get_stats(), dict)


@pytest.mark.asyncio
async def test_load_and_get_missing(tmp_path):
    store = ArtifactStore(base_path=str(tmp_path))
    assert await store.load_artifact("missing") is None
    assert await store.get_artifact("missing") is None
    assert await store.delete_artifact("missing") is False
