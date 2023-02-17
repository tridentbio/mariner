"""
Entrypoint to run the mariner API
"""
import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run("api.fastapi_app:app", host="0.0.0.0", port=80, reload=bool(os.getenv('RESTART')))

