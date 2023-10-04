"""
Entrypoint to run the mariner API
"""
import logging
import os
import urllib.parse

import ray
import uvicorn

from mariner.core.config import get_app_settings

LOG = logging.getLogger(__name__)

if __name__ == "__main__":
    url = urllib.parse.urlparse(get_app_settings("server").host)
    host = "0.0.0.0"
    port = url.port or 80
    LOG.warning("Started uvicorn app at %s:%d", host, port)
    ray.init(address="auto", allow_multiple=True)
    uvicorn.run(
        "api.fastapi_app:app",
        host=host,
        port=port,
        log_config="./log_config.json",
        reload=bool(os.getenv("RESTART")),
    )
