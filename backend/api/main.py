"""
Entrypoint to run the mariner API
"""
import os
import urllib.parse

import uvicorn

from mariner.core.config import get_app_settings

if __name__ == "__main__":
    url = urllib.parse.urlparse(get_app_settings("server").host)
    uvicorn.run(
        "api.fastapi_app:app",
        host="0.0.0.0",
        port=url.port or 80,
        reload=bool(os.getenv("RESTART")),
    )
