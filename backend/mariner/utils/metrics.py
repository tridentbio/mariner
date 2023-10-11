from prometheus_client import Summary

REQUEST_TIME = Summary(
    "request_processing_seconds",
    "Time spent processing request",
    labelnames=["endpoint", "method"],
)
