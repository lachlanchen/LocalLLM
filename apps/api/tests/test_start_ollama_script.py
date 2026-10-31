from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path: Path):
    root = tmp_path / "app with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).parents[3] / "scripts"
    for name in ("start-ollama.sh", "install-ollama-local.sh"):
        shutil.copy2(source / name, scripts / name)
    binary = tmp_path / "fake-bin"
    binary.mkdir()
    uname = binary / "uname"
    uname.write_text('#!/bin/sh\ncase "$1" in -s) echo "$TEST_OS";; -m) echo "$TEST_ARCH";; esac\n')
    uname.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OLLAMA_", "LOCALLLM_"))}
    env.update(PATH=f"{binary}:{os.environ['PATH']}", TEST_OS="Darwin", TEST_ARCH="arm64")
    return root, env


def fake_ollama(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nprintf "models=%s\\nhost=%s\\nargs=%s\\ncuda=%s\\n" "$OLLAMA_MODELS" "$OLLAMA_HOST" "$*" "${CUDA_VISIBLE_DEVICES:-}"\n')
    path.chmod(0o755)


def run(root: Path, env: dict[str, str], script: str = "start-ollama.sh"):
    return subprocess.run(["bash", str(root / "scripts" / script)], env=env, text=True,
                          capture_output=True, timeout=10, check=False)


def test_darwin_reuses_native_binary_and_existing_model_store(launcher) -> None:
    root, env = launcher
    fake_ollama(root / ".local/ollama/ollama")
    (root / "models").mkdir()
    result = run(root, env)
    assert result.returncode == 0, result.stderr
    assert f"models={root}/models\n" in result.stdout
    assert "host=127.0.0.1:11434\nargs=serve\n" in result.stdout
    assert not (root / ".local/models/ollama").exists()


def test_explicit_binary_and_model_store_are_respected(launcher) -> None:
    root, env = launcher
    binary = root / "custom ollama"
    fake_ollama(binary)
    env.update(LOCALLLM_OLLAMA_BIN=str(binary), OLLAMA_MODELS=str(root / "custom models"),
               OLLAMA_HOST="127.0.0.1:11435")
    result = run(root, env)
    assert result.returncode == 0, result.stderr
    assert f"models={root}/custom models\nhost=127.0.0.1:11435\n" in result.stdout


def test_missing_explicit_binary_does_not_install_or_download(launcher) -> None:
    root, env = launcher
    env.update(TEST_OS="Linux", TEST_ARCH="x86_64", LOCALLLM_OLLAMA_BIN=str(root / "missing"))
    result = run(root, env)
    assert result.returncode == 1
    assert "set LOCALLLM_OLLAMA_BIN" in result.stderr
    assert not (root / ".local").exists()


def test_linux_defaults_and_gpu_selection_are_preserved(launcher) -> None:
    root, env = launcher
    fake_ollama(root / ".local/ollama/bin/ollama")
    env.update(TEST_OS="Linux", TEST_ARCH="x86_64", LOCALLLM_OLLAMA_CUDA_VISIBLE_DEVICES="0")
    result = run(root, env)
    assert result.returncode == 0, result.stderr
    assert f"models={root}/.local/models/ollama\n" in result.stdout
    assert "cuda=0\n" in result.stdout


@pytest.mark.parametrize("platform,arch", [("Darwin", "arm64"), ("Linux", "aarch64")])
def test_pinned_linux_installer_refuses_other_platforms_before_writes(launcher, platform, arch) -> None:
    root, env = launcher
    env.update(TEST_OS=platform, TEST_ARCH=arch)
    result = run(root, env, "install-ollama-local.sh")
    assert result.returncode == 1
    assert "Linux x86_64 only" in result.stderr
    assert not (root / ".local").exists()
