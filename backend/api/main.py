"""
Entrypoint to run the mariner API
"""
import os

import uvicorn

from fleet.options import options_manager

if __name__ == "__main__":
    options_manager.import_libs()
    uvicorn.run(
        "api.fastapi_app:app",
        host="0.0.0.0",
        port=80,
        reload=bool(os.getenv("RESTART")),
    )
