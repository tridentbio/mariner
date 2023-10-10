"""
Entrypoint to run the mariner API
"""
import logging
import logging.config
import os
import urllib.parse

import uvicorn

from mariner.core.config import get_app_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s    🚢 %(levelname)s %(filename)s:%(lineno)d - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOG = logging.getLogger("mariner.api")

if __name__ == "__main__":
    url = urllib.parse.urlparse(get_app_settings("server").host)
    host = "0.0.0.0"
    port = url.port or 80
    LOG.warning("🚀 Starting uvicorn app at %s:%d", host, port)
    uvicorn.run(
        "api.fastapi_app:app",
        host=host,
        port=port,
        log_config="./log_config.json",
        log_level="debug",
        use_colors=True,
        reload=bool(os.getenv("RESTART")),
    )
