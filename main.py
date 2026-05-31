"""Main entry point for the Multi-Agent RCA System."""

import sys
import uvicorn

from config.settings import get_settings

settings = get_settings()

if __name__ == "__main__":
    # Disable reload in Windows non-TTY environments to prevent uvicorn subprocess stdin duplication crash
    is_windows = sys.platform == "win32"
    stdin_is_tty = sys.stdin.isatty() if sys.stdin else False
    should_reload = settings.app_debug and not (is_windows and not stdin_is_tty)

    uvicorn.run(
        "services.api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=should_reload,
        workers=1,
    )
