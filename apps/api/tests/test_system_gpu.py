import json
from unittest.mock import AsyncMock

import pytest

from localllm import system

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("metal_key,metal_value", [
    ("spdisplays_mtlgpufamilysupport", "spdisplays_metal4"),
    ("spdisplays_metal", "spdisplays_supported"),
])
async def test_mac_reports_observed_metal_without_inventing_vram(monkeypatch, metal_key, metal_value):
    monkeypatch.setattr(system.sys, "platform", "darwin")
    command = AsyncMock(return_value=(0, json.dumps({"SPDisplaysDataType": [
        {"sppci_model": "Apple M5 Pro", metal_key: metal_value}
    ]})))
    monkeypatch.setattr(system, "_command", command)
    result = await system.gpu_status()
    assert result["ok"] is True
    assert result["backend"] == "metal"
    assert result["devices"][0]["name"] == "Apple M5 Pro"
    assert "memory_total_mb" not in result["devices"][0]
    assert "separately" in result["diagnosis"]
    command.assert_awaited_once_with("system_profiler", "SPDisplaysDataType", "-json")


@pytest.mark.parametrize("reply", [
    (124, "timed out"), (0, "invalid json"), (0, "[]"),
    (0, '{"SPDisplaysDataType":[{"spdisplays_metal":"spdisplays_unsupported"}]}'),
    (0, '{"SPDisplaysDataType":[]}'),
])
async def test_mac_unknown_hardware_never_claims_ready(monkeypatch, reply):
    monkeypatch.setattr(system.sys, "platform", "darwin")
    monkeypatch.setattr(system, "_command", AsyncMock(return_value=reply))
    result = await system.gpu_status()
    assert result["ok"] is False
    assert result["devices"] == []
    assert "could not be verified" in result["diagnosis"]


async def test_linux_nvidia_inventory_is_preserved(monkeypatch):
    monkeypatch.setattr(system.sys, "platform", "linux")
    monkeypatch.setattr(system, "_command", AsyncMock(return_value=(0, "0, RTX 4090, 24564, 100, 24464, 40, 60.1")))
    result = await system.gpu_status()
    assert result["ok"] is True
    assert result["devices"][0]["memory_free_mb"] == 24464


async def test_linux_driver_mismatch_remains_unhealthy(monkeypatch):
    monkeypatch.setattr(system.sys, "platform", "linux")
    monkeypatch.setattr(system, "_command", AsyncMock(side_effect=[
        (1, "Failed to initialize NVML: Driver/library version mismatch"), (0, "test-version")
    ]))
    result = await system.gpu_status()
    assert result["ok"] is False
    assert "mismatch" in result["diagnosis"]
