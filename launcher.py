"""
Pipeline Optima™ — Standalone Launcher
Starts the Streamlit app locally and opens it in the system browser.
Works both as a plain Python script and as a PyInstaller-packaged .exe.
"""
import os
import sys
import subprocess
import threading
import time
import webbrowser

PORT = 8501


def _get_base_dir():
    """Return the directory containing the app files (handles PyInstaller _MEIPASS)."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _start_streamlit(base_dir):
    app_path = os.path.join(base_dir, 'pipeline_optimization_app.py')
    streamlit_config = os.path.join(base_dir, '.streamlit', 'config.toml')

    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_path,
        f'--server.port={PORT}',
        '--server.headless=true',
        '--server.enableCORS=false',
        '--server.enableXsrfProtection=false',
        '--browser.gatherUsageStats=false',
        '--theme.base=dark',
    ]
    if os.path.isfile(streamlit_config):
        cmd += ['--global.dataFrameSerialization=legacy']

    subprocess.Popen(cmd, cwd=base_dir)


def main():
    base_dir = _get_base_dir()

    t = threading.Thread(target=_start_streamlit, args=(base_dir,), daemon=True)
    t.start()

    # Wait for Streamlit to be ready before opening the browser
    time.sleep(5)
    webbrowser.open(f'http://localhost:{PORT}')

    print(f"\nPipeline Optima is running at http://localhost:{PORT}")
    print("Close this window to stop the application.\n")

    try:
        # Keep the process alive until the user closes the window
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Pipeline Optima...")


if __name__ == '__main__':
    main()
