"""
Pipeline Optima™ — desktop launcher.

Starts the FastAPI server on localhost:8765 and opens the default browser
once the server is confirmed to be listening. Entry point for PyInstaller.

IMPORTANT: stdout/stderr redirect must happen before ANY other import
because uvicorn's logging formatter calls stream.isatty() on startup.
"""

# ── 1. Redirect stdout/stderr immediately (windowed PyInstaller has them as None) ──
import os
import sys

_log_path = os.path.join(os.path.expanduser("~"), "PipelineOptima.log")
try:
    _logfile = open(_log_path, "w", buffering=1, encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _logfile
    if sys.stderr is None:
        sys.stderr = _logfile
except Exception:
    pass

# ── 2. Standard imports ────────────────────────────────────────────────────────────
import socket
import threading
import time
import traceback
import webbrowser

PORT = int(os.environ.get("OPTIMA_PORT", "8765"))
HOST = "127.0.0.1"

_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)
os.chdir(_base)

# ── 3. Safe uvicorn log config (plain Formatter, no isatty calls) ─────────────────
_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "logging.Formatter",
            "fmt": "%(levelname)s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        },
    },
    "loggers": {
        "uvicorn":        {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "uvicorn.error":  {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
    },
}


def _wait_then_open_browser() -> None:
    """Poll the port until the server is accepting connections, then open browser."""
    for _ in range(60):          # up to 30 seconds
        try:
            conn = socket.create_connection((HOST, PORT), timeout=0.5)
            conn.close()
            webbrowser.open(f"http://{HOST}:{PORT}")
            return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)


def _show_error(msg: str) -> None:
    """Show a GUI error dialog so the user sees what went wrong."""
    try:
        import tkinter as tk
        import tkinter.messagebox as mb
        root = tk.Tk()
        root.withdraw()
        mb.showerror(
            "Pipeline Optima — Startup Error",
            f"{msg}\n\nFull log: {_log_path}",
        )
        root.destroy()
    except Exception:
        pass  # if tkinter unavailable, error is at least in the log file


if __name__ == "__main__":
    try:
        threading.Thread(target=_wait_then_open_browser, daemon=True).start()

        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host=HOST,
            port=PORT,
            log_config=_LOG_CONFIG,
        )
    except Exception:
        err = traceback.format_exc()
        try:
            print(err, flush=True)
        except Exception:
            pass
        _show_error(f"Server failed to start:\n\n{err[:600]}")
        sys.exit(1)
