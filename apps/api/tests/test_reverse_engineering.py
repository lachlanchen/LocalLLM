from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from localllm.config import Settings
from localllm.reverse_engineering import _exec, delete_inspection, inspect_upload


@pytest.mark.asyncio
async def test_binary_inspection_is_static_and_hashed(tmp_path) -> None:
    sample = b"MZ\x00\x00This is only a static inspection test string\x00"
    upload = UploadFile(filename="sample.exe", file=BytesIO(sample))
    settings = Settings(data_dir=tmp_path)
    result = await inspect_upload(upload, settings)

    assert result["filename"] == "sample.exe"
    assert result["size"] == len(sample)
    assert len(result["sha256"]) == 64
    assert "never executed" in result["safety"]
    assert any("static inspection" in value for value in result["strings"])
    assert "stored_path" not in result
    assert (tmp_path / "reverse" / "uploads" / f"{result['id']}.json").stat().st_mode & 0o777 == 0o600

    deleted = await delete_inspection(result["id"], settings)
    assert deleted == {"deleted": True, "id": result["id"]}
    assert not list((tmp_path / "reverse" / "uploads").iterdir())


@pytest.mark.asyncio
async def test_delete_recovers_binary_when_metadata_is_missing(tmp_path) -> None:
    upload = UploadFile(filename="sample.bin", file=BytesIO(b"static evidence"))
    settings = Settings(data_dir=tmp_path)
    result = await inspect_upload(upload, settings)
    directory = tmp_path / "reverse" / "uploads"
    (directory / f"{result['id']}.json").unlink()

    deleted = await delete_inspection(result["id"], settings)

    assert deleted == {"deleted": True, "id": result["id"]}
    assert not list(directory.iterdir())


@pytest.mark.asyncio
async def test_command_timeout_terminates_child() -> None:
    code, output, truncated = await _exec(
        "python3", "-c", "import time; time.sleep(30)", timeout=0.01
    )

    assert code == 124
    assert "timed out" in output
    assert truncated is True
