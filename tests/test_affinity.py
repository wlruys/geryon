from __future__ import annotations

from geryon.executors import affinity


def test_discover_core_pool_prefers_explicit_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CORES", "8,9")
    monkeypatch.setattr(affinity, "_discover_psutil_core_pool", lambda: [0, 1, 2, 3])
    assert affinity.discover_core_pool("4,6") == [4, 6]


def test_discover_core_pool_uses_psutil_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CORES", raising=False)
    monkeypatch.setattr(affinity, "_discover_psutil_core_pool", lambda: [2, 4, 6])
    assert affinity.discover_core_pool(None) == [2, 4, 6]
