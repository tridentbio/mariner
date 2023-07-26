"""
Entrypoint to run the mariner API
"""
import os
import urllib.parse

import uvicorn

from fleet.options import options_manager
from mariner.core.config import get_app_settings

if __name__ == "__main__":
    options_manager.import_libs()
    url = urllib.parse.urlparse(get_app_settings("server").host)
    uvicorn.run(
        "api.fastapi_app:app",
        host=url.hostname or "0.0.0.0",
        port=url.port or 80,
        reload=bool(os.getenv("RESTART")),
    )
