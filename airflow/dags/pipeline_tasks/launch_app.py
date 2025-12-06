import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path(__file__).resolve().parents[3]
APP_PATH = BASE_DIR / "app.py"


def run() -> None:
    """Launch Streamlit app. This will start a background process."""
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Streamlit app not found at {APP_PATH}")
    
    logger.info("Launching Streamlit app at %s", APP_PATH)
    
    # Start Streamlit in background
    cmd = [
        "streamlit", "run", str(APP_PATH),
        "--server.port", "8501",
        "--server.headless", "true",
    ]
    
    try:
        # Run in background - don't wait for it to finish
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(BASE_DIR),
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        if process.poll() is None:
            logger.info("✅ Streamlit app started successfully on http://localhost:8501")
            logger.info("Process PID: %d", process.pid)
        else:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Streamlit failed to start: {stderr.decode()}")
            
    except FileNotFoundError:
        raise RuntimeError("Streamlit not installed. Run: pip install streamlit")


if __name__ == "__main__":  # pragma: no cover
    run()
