from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from localllm.config import Settings
from localllm.reverse_engineering import inspect_upload


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

