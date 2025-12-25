"""Desktop launcher that embeds the Streamlit UI in a native window."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from multiprocessing import freeze_support
from pathlib import Path

import webview


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.3)


def _run_streamlit(app_path: Path, port: int) -> None:
    from streamlit.web import bootstrap

    bootstrap.run(
        str(app_path),
        False,
        [],
        {
            "server.headless": True,
            "server.port": port,
            "server.address": "127.0.0.1",
            "browser.serverAddress": "127.0.0.1",
            "browser.gatherUsageStats": False,
        },
    )


def main() -> int:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app_path = base_dir / "pipeline_optimization_app.py"
    if not app_path.exists():
        raise FileNotFoundError(
            "pipeline_optimization_app.py not found in bundle. "
            "Rebuild with the build script to include it."
        )

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=_run_streamlit, args=(app_path, port), daemon=True)
    thread.start()

    _wait_for_server(url)

    def _shutdown() -> None:
        os._exit(0)

    webview.create_window("Pipeline Optimization", url, width=1400, height=900)
    webview.start(gui=None, debug=False, on_closed=_shutdown)

    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
