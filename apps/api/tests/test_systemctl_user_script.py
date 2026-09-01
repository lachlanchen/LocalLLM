from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path


def test_systemctl_user_uses_canonical_runtime_bus(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[3]
    wrapper = root / "scripts" / "systemctl-user.sh"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    bus_path = runtime / "bus"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_systemctl = fake_bin / "systemctl"
    fake_systemctl.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'address=%s\\n' \"$DBUS_SESSION_BUS_ADDRESS\"\n"
        "printf 'runtime=%s\\n' \"$XDG_RUNTIME_DIR\"\n"
        "printf 'args=%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_systemctl.chmod(0o755)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(bus_path))
    listener.listen(1)
    try:
        result = subprocess.run(
            [str(wrapper), "show", "lazyedge-tunnel.service"],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "XDG_RUNTIME_DIR": str(runtime),
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/wrong/session-bus",
            },
        )
    finally:
        listener.close()

    assert result.returncode == 0, result.stderr
    assert f"address=unix:path={bus_path}" in result.stdout
    assert f"runtime={runtime}" in result.stdout
    assert "args=--user show lazyedge-tunnel.service" in result.stdout


def test_systemctl_user_fails_closed_without_runtime_bus(tmp_path: Path) -> None:
    root = Path(__file__).parents[3]
    wrapper = root / "scripts" / "systemctl-user.sh"
    result = subprocess.run(
        [str(wrapper), "is-active", "localllm-api.service"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "LOCALLLM_USER_RUNTIME_DIR": str(tmp_path / "missing"),
        },
    )

    assert result.returncode != 0
    assert "canonical user bus is unavailable" in result.stderr
