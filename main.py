"""Main entry point for the Multi-Agent RCA System."""

import uvicorn

from config.settings import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "services.api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        workers=1,
    )
