import uvicorn
import sys


if __name__ == "__main__":
    uvicorn.run("api.fastapi_app:app", host="0.0.0.0", port=80)
