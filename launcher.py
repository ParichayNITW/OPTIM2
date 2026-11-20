"""
Pipeline Optima™ — desktop launcher.

Opens a native Qt6 window (no browser).  The FastAPI server runs silently
in a background thread; QWebEngineView renders the React UI inside the window.

Entry point for PyInstaller.

IMPORTANT: stdout/stderr redirect MUST happen before any other import because
uvicorn's logging formatter calls stream.isatty() on startup in windowed mode.
"""

# ── 1. Redirect stdout/stderr immediately (windowed PyInstaller sets them None) ──
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

# ── 2. Set writable data directory BEFORE any import reads DATABASE_URL ──────────
# On Windows: %APPDATA%\Pipeline Optima\  (always writable, persists across restarts)
if sys.platform == "win32":
    _appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    _data_dir = os.path.join(_appdata, "Pipeline Optima")
else:
    _data_dir = os.path.join(os.path.expanduser("~"), ".pipeline-optima")

try:
    os.makedirs(_data_dir, exist_ok=True)
except Exception:
    _data_dir = os.path.expanduser("~")

_db_path = os.path.join(_data_dir, "optima.db")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_path}")

# ── 3. Standard library imports ───────────────────────────────────────────────────
import socket
import threading
import time
import traceback

PORT = int(os.environ.get("OPTIMA_PORT", "8765"))
HOST = "127.0.0.1"
APP_URL = f"http://{HOST}:{PORT}"

_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
if _base not in sys.path:
    sys.path.insert(0, _base)
os.chdir(_base)

# ── 4. Safe uvicorn log config (plain Formatter avoids isatty crash) ─────────────
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


def _start_server() -> None:
    """Run uvicorn in this thread (called from a daemon thread)."""
    try:
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


def _wait_for_server(timeout: float = 30.0) -> bool:
    """Block until the server accepts connections or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = socket.create_connection((HOST, PORT), timeout=0.5)
            conn.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


# ── 5. Qt desktop window ──────────────────────────────────────────────────────────

def main():
    # Start the FastAPI server in the background
    srv_thread = threading.Thread(target=_start_server, daemon=True)
    srv_thread.start()

    # Import Qt (heavy — do after server thread starts to parallelise startup)
    from PySide6.QtCore import QUrl, Qt, QTimer
    from PySide6.QtGui import QIcon, QPixmap, QColor, QPainter
    from PySide6.QtWidgets import QApplication, QMainWindow, QSplashScreen, QLabel
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage

    app = QApplication(sys.argv)
    app.setApplicationName("Pipeline Optima")
    app.setOrganizationName("Pipeline Optima")

    # ── App icon ──────────────────────────────────────────────────────────────────
    icon_path = os.path.join(_base, "assets", "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # ── Splash screen ─────────────────────────────────────────────────────────────
    splash_pix = QPixmap(480, 280)
    splash_pix.fill(QColor("#1e3a5f"))
    painter = QPainter(splash_pix)
    painter.setPen(QColor("#ffffff"))
    from PySide6.QtGui import QFont
    font = QFont("Segoe UI", 22, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(splash_pix.rect(), Qt.AlignmentFlag.AlignCenter, "Pipeline Optima™")
    font2 = QFont("Segoe UI", 11)
    painter.setFont(font2)
    painter.setPen(QColor("#93c5fd"))
    from PySide6.QtCore import QRect
    painter.drawText(
        QRect(0, 170, 480, 50),
        Qt.AlignmentFlag.AlignCenter,
        "Starting optimization engine…",
    )
    painter.end()
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()

    # ── Wait for server ───────────────────────────────────────────────────────────
    ready = _wait_for_server(timeout=30.0)
    splash.close()

    if not ready:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(
            None,
            "Pipeline Optima — Startup Error",
            f"The optimization engine failed to start within 30 seconds.\n\nCheck log: {_log_path}",
        )
        sys.exit(1)

    # ── Main window ───────────────────────────────────────────────────────────────
    window = QMainWindow()
    window.setWindowTitle("Pipeline Optima™")
    window.resize(1400, 900)
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))

    # Centre on screen
    screen = app.primaryScreen().geometry()
    x = (screen.width() - 1400) // 2
    y = (screen.height() - 900) // 2
    window.move(max(0, x), max(0, y))

    # ── Embedded browser ──────────────────────────────────────────────────────────
    view = QWebEngineView()
    settings = view.settings()
    settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.AllowWindowActivationFromJavaScript, True)

    # Inject a custom user-agent so the app can detect it's the desktop build
    page = view.page()
    profile = page.profile()
    profile.setHttpUserAgent("PipelineOptimaDesktop/1.0 Qt6")

    view.setUrl(QUrl(APP_URL))
    window.setCentralWidget(view)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
