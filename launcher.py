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
# Ensure the solver directory is on sys.path.
# In a PyInstaller bundle, sys._MEIPASS is the extraction directory.
# When running from source, this file lives in the project root.
# ---------------------------------------------------------------------------
_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)

# Change working directory so relative paths (SQLite, frontend/dist) resolve correctly
os.chdir(_base)


def _open_browser() -> None:
    time.sleep(2.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Pipeline Optima™ starting on http://{HOST}:{PORT} ...")
    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        log_level="warning",
    )
