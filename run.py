"""
Root entry point to launch the Python Neural Snake backend and web dashboard.
"""
import sys
import webbrowser
import uvicorn

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if __name__ == "__main__":
    port = 8000
    url = f"http://localhost:{port}"

    print("=" * 60)
    print("  [+] NEURAL SNAKE // PYTHON DEEP Q-LEARNING DASHBOARD")
    print("=" * 60)
    print(f"  Server starting at: {url}")
    print("  Serving Web Interface with Live WebSocket Telemetry...")
    print("=" * 60)

    # Open browser automatically
    webbrowser.open(url)

    # Launch server
    uvicorn.run("backend.server:app", host="127.0.0.1", port=port, log_level="info")

