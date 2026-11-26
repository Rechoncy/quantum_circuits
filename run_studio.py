"""
Quantum Studio v2 Desktop Launcher (React + Tailwind + Plotly + FastAPI + pywebview / Edge App).

Usage:
  1. Double-click `QuantumStudio.exe` or `start_studio.bat`
  2. Native Desktop Window (using embedded Python):
       .\\python_env\\python.exe run_studio.py
  3. Local Web Server Mode (for browser access / debugging):
       .\\python_env\\python.exe run_studio.py --web --port 8765
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ICON_ICO_PATH = ROOT_DIR / "assets" / "quantum_studio.ico"

# When running under pythonw.exe (Windows GUI subsystem), sys.stdout and sys.stderr are None.
# Provide a safe non-TTY stream so uvicorn, logging, and print() never crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# Ensure embedded python_env DLL directories are registered automatically
_EMBED_ENV = ROOT_DIR / "python_env"
_EMBED_BIN = _EMBED_ENV / "Library" / "bin"
if _EMBED_BIN.is_dir():
    os.environ["PATH"] = f"{_EMBED_ENV};{_EMBED_BIN};{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(_EMBED_ENV))
            os.add_dll_directory(str(_EMBED_BIN))
        except Exception:
            pass

os.environ.setdefault("MPMATH_NOGMPY", "1")

import uvicorn

from quantum_solver.studio.server import create_app


def _set_windows_app_id() -> None:
    """Register distinct Windows AppUserModelID so Taskbar displays our custom icon."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("QuantumStudio.Desktop.v2")
    except Exception:
        pass


def _apply_win32_window_icon(window_title: str, ico_path: Path) -> None:
    """Apply custom .ico to the native Win32 window titlebar and taskbar via WM_SETICON."""
    if sys.platform != "win32" or not ico_path.is_file():
        return
    try:
        user32 = ctypes.windll.user32
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1

        h_icon_big = user32.LoadImageW(
            0, str(ico_path), IMAGE_ICON, 48, 48, LR_LOADFROMFILE
        )
        h_icon_small = user32.LoadImageW(
            0, str(ico_path), IMAGE_ICON, 16, 16, LR_LOADFROMFILE | LR_DEFAULTSIZE
        )
        if not h_icon_big and not h_icon_small:
            return

        for _ in range(30):
            hwnd = user32.FindWindowW(None, window_title)
            if hwnd:
                if h_icon_big:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_icon_big)
                if h_icon_small:
                    user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_icon_small)
                user32.SetForegroundWindow(hwnd)
                break
            time.sleep(0.15)
    except Exception:
        pass


def _find_free_port(preferred: int = 8765) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(port: int, timeout: float = 12.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _run_chromium_app_window(url: str) -> None:
    """Open a dedicated chromeless Edge/Chrome desktop app window and wait until closed."""
    candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    profile_dir = Path(tempfile.gettempdir()) / "QuantumStudio_AppProfile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    for exe in candidates:
        if exe and os.path.isfile(exe):
            try:
                proc = subprocess.Popen(
                    [
                        exe,
                        f"--app={url}",
                        "--window-size=1480,920",
                        f"--user-data-dir={profile_dir}",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ]
                )
                proc.wait()
                return
            except Exception:
                continue

    webbrowser.open(url)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main() -> int:
    _set_windows_app_id()
    parser = argparse.ArgumentParser(description="Launch Quantum Studio v2 Desktop App")
    parser.add_argument("--web", action="store_true", help="Run as local HTTP server instead of desktop window")
    parser.add_argument("--open-browser", action="store_true", help="Open default browser automatically in --web mode")
    parser.add_argument("--port", type=int, default=8765, help="Local server port (default: 8765)")
    args = parser.parse_args()

    port = args.port if args.web else _find_free_port(args.port)
    app = create_app()

    if args.web:
        url = f"http://127.0.0.1:{port}"
        print(f"Quantum Studio v2 running at {url}")
        if args.open_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", log_config=None)
        return 0

    # Launch FastAPI in a daemon thread
    server_config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", log_config=None
    )
    server = uvicorn.Server(server_config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    _wait_for_server(port, timeout=12.0)
    url = f"http://127.0.0.1:{port}"
    win_title = "Quantum Studio — 量子线路设计与多后端演化工作台"

    webview_failed = {"failed": False}
    try:
        import webview
        from webview.platforms import edgechromium

        orig_ready = edgechromium.EdgeChrome.on_webview_ready

        def _checked_webview_ready(self_ec, sender, wv_args):
            if not wv_args.IsSuccess:
                webview_failed["failed"] = True
                try:
                    self_ec.form.Close()
                except Exception:
                    pass
                return
            return orig_ready(self_ec, sender, wv_args)

        edgechromium.EdgeChrome.on_webview_ready = _checked_webview_ready

        webview.settings["ALLOW_DOWNLOADS"] = True

        class DesktopBridge:
            def __init__(self) -> None:
                self.window: object = None

            def save_file(self, filename: str, content: str, is_base64: bool = False) -> dict:
                if self.window is None:
                    return {"saved": False, "error": "window_not_ready"}
                ext = Path(filename).suffix.lower()
                type_map = {
                    ".svg": ("SVG 矢量图 (*.svg)", "All files (*.*)"),
                    ".csv": ("CSV 数据表 (*.csv)", "All files (*.*)"),
                    ".json": ("JSON 工程文件 (*.json)", "All files (*.*)"),
                    ".md": ("Markdown 报告 (*.md)", "All files (*.*)"),
                    ".qasm": ("OpenQASM 量子代码 (*.qasm)", "All files (*.*)"),
                    ".tex": ("LaTeX Quantikz 代码 (*.tex)", "All files (*.*)"),
                    ".png": ("PNG 图像 (*.png)", "All files (*.*)"),
                }
                ftypes = type_map.get(ext, ("All files (*.*)",))
                res = self.window.create_file_dialog(  # type: ignore[attr-defined]
                    webview.SAVE_DIALOG,
                    save_filename=filename,
                    file_types=ftypes,
                )
                if not res:
                    return {"saved": False, "cancelled": True}
                chosen = res[0] if isinstance(res, (list, tuple)) else str(res)
                if not chosen:
                    return {"saved": False, "cancelled": True}
                out_path = Path(chosen)
                if ext and not out_path.suffix:
                    out_path = out_path.with_suffix(ext)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if is_base64:
                    out_path.write_bytes(base64.b64decode(content))
                else:
                    out_path.write_text(content, encoding="utf-8")
                return {"saved": True, "path": str(out_path)}

        bridge = DesktopBridge()
        win = webview.create_window(
            title=win_title,
            url=url,
            js_api=bridge,
            width=1480,
            height=920,
            min_size=(1100, 700),
            background_color="#0E1116",
        )
        bridge.window = win
        if ICON_ICO_PATH.is_file():
            threading.Thread(
                target=_apply_win32_window_icon,
                args=(win_title, ICON_ICO_PATH),
                daemon=True,
            ).start()
        webview.start(icon=str(ICON_ICO_PATH) if ICON_ICO_PATH.is_file() else None)
    except Exception:
        webview_failed["failed"] = True

    if webview_failed["failed"]:
        _run_chromium_app_window(url)

    server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
