from pathlib import Path

from localllm.system import storage_status


def test_storage_status_has_nonnegative_values(tmp_path: Path) -> None:
    status = storage_status(tmp_path)
    assert status["total"] > 0
    assert status["free"] >= 0
