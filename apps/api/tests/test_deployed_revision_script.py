from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def test_deployed_revision_gate_accepts_only_the_exact_clean_worktree(tmp_path: Path) -> None:
    root = Path(__file__).parents[3]
    gate = root / "scripts" / "verify-deployed-revision.sh"
    repository = tmp_path / "repository"
    repository.mkdir()
    assert run("git", "init", "--quiet", cwd=repository).returncode == 0
    assert run("git", "config", "user.name", "LocalLLM test", cwd=repository).returncode == 0
    assert run("git", "config", "user.email", "localllm-test@example.invalid", cwd=repository).returncode == 0
    tracked = repository / "tracked.txt"
    tracked.write_text("sealed\n", encoding="utf-8")
    assert run("git", "add", "tracked.txt", cwd=repository).returncode == 0
    assert run("git", "commit", "--quiet", "-m", "sealed", cwd=repository).returncode == 0
    revision = run("git", "rev-parse", "HEAD", cwd=repository).stdout.strip()

    accepted = run(str(gate), str(repository), revision, cwd=repository)
    assert accepted.returncode == 0
    assert accepted.stdout == ""

    wrong_revision = "0" * 40 if revision != "0" * 40 else "1" * 40
    wrong = run(str(gate), str(repository), wrong_revision, cwd=repository)
    assert wrong.returncode != 0
    assert "does not match" in wrong.stderr

    tracked.write_text("mutated\n", encoding="utf-8")
    dirty = run(str(gate), str(repository), revision, cwd=repository)
    assert dirty.returncode != 0
    assert "not clean" in dirty.stderr

    tracked.write_text("sealed\n", encoding="utf-8")
    (repository / "untracked.txt").write_text("unexpected\n", encoding="utf-8")
    untracked = run(str(gate), str(repository), revision, cwd=repository)
    assert untracked.returncode != 0
    assert "not clean" in untracked.stderr


def test_private_credential_gate_rejects_replaceable_or_aliased_sources(tmp_path: Path) -> None:
    root = Path(__file__).parents[3]
    gate = root / "scripts" / "verify-private-credential-source.sh"
    secure = tmp_path / "secure"
    secure.mkdir(mode=0o700)
    credential = secure / "search-key"
    credential.write_text("test-credential", encoding="ascii")
    credential.chmod(0o600)

    accepted = run(str(gate), str(credential), cwd=root)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout == ""

    credential.chmod(0o640)
    exposed = run(str(gate), str(credential), cwd=root)
    assert exposed.returncode != 0
    assert "readable only by its owner" in exposed.stderr
    credential.chmod(0o600)

    alias = secure / "search-key-link"
    alias.symlink_to(credential)
    symlinked = run(str(gate), str(alias), cwd=root)
    assert symlinked.returncode != 0
    assert "must not traverse symlinks or aliases" in symlinked.stderr

    hardlink = secure / "search-key-hardlink"
    hardlink.hardlink_to(credential)
    linked = run(str(gate), str(credential), cwd=root)
    assert linked.returncode != 0
    assert "single-link regular file" in linked.stderr
    hardlink.unlink()

    secure.chmod(0o707)
    replaceable = run(str(gate), str(credential), cwd=root)
    assert replaceable.returncode != 0
    assert "writable by another account" in replaceable.stderr
    secure.chmod(0o700)

    if shutil.which("setfacl") is not None:
        acl = run("setfacl", "-m", "u:nobody:rwx", str(secure), cwd=root)
        assert acl.returncode == 0, acl.stderr
        extended_acl = run(str(gate), str(credential), cwd=root)
        assert extended_acl.returncode != 0
        assert "extended access ACLs" in extended_acl.stderr


def test_rendered_api_unit_pins_release_and_credential_after_environment_file(
    tmp_path: Path,
) -> None:
    if shutil.which("systemd-analyze") is None:
        return

    root = Path(__file__).parents[3]
    template = (root / "deploy" / "systemd" / "localllm-api.service.in").read_text()
    revision = "a" * 40
    release_id = f"{'a' * 8}-{'b' * 8}"
    credential = tmp_path / "search-key"
    credential.write_text("test-credential", encoding="ascii")
    credential.chmod(0o600)
    rendered = template
    replacements = {
        "@PROJECT_ROOT@": str(root),
        "@SEARCH_API_LOAD_CREDENTIAL@": (
            f"LoadCredential=localllm-search-api-key:{credential}"
        ),
        "@SEARCH_API_UNSET_ENVIRONMENT@": (
            "UnsetEnvironment=LOCALLLM_SEARCH_API_KEY "
            "LOCALLLM_SEARCH_API_KEY_FILE CREDENTIALS_DIRECTORY"
        ),
        "@SEARCH_API_EXEC_ENVIRONMENT@": (
            "CREDENTIALS_DIRECTORY=%d "
            "LOCALLLM_SEARCH_API_KEY_FILE=%d/localllm-search-api-key"
        ),
        "@DEPLOYED_RELEASE_ID@": release_id,
        "@DEPLOYED_REVISION@": revision,
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    assert "@" not in rendered
    assert "EnvironmentFile=-" in rendered
    assert "UnsetEnvironment=LOCALLLM_RELEASE_ID" in rendered
    assert "UnsetEnvironment=LOCALLLM_SEARCH_API_KEY" in rendered
    assert f"/usr/bin/env LOCALLLM_RELEASE_ID={release_id}" in rendered
    assert "CREDENTIALS_DIRECTORY=%d" in rendered
    assert "LOCALLLM_SEARCH_API_KEY_FILE=%d/localllm-search-api-key" in rendered

    unit = tmp_path / "localllm-api-render-test.service"
    unit.write_text(rendered, encoding="utf-8")
    verified = run("systemd-analyze", "--user", "verify", str(unit), cwd=root)
    assert verified.returncode == 0, verified.stderr
