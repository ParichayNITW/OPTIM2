"""
Pipeline Optima™ — desktop launcher.

Starts the FastAPI server on localhost:8765 and opens the default browser
after a short delay. This is the entry point bundled by PyInstaller.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser

PORT = int(os.environ.get("OPTIMA_PORT", "8765"))
HOST = "127.0.0.1"

# ---------------------------------------------------------------------------
# In windowed mode (no console), sys.stdout/stderr are None.
# Redirect them to a log file so uvicorn logging doesn't crash on isatty().
# ---------------------------------------------------------------------------
_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

os.chdir(_base)

_log_path = os.path.join(os.path.expanduser("~"), "PipelineOptima.log")
if sys.stdout is None:
    sys.stdout = open(_log_path, "w", buffering=1, encoding="utf-8")
if sys.stderr is None:
    sys.stderr = sys.stdout


def _open_browser() -> None:
    time.sleep(2.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        log_config=None,   # skip uvicorn logging config (avoids isatty crash)
    )
