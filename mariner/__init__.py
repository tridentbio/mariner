"""
The mariner package implements business rules regarding users (users.py)
datasets (datasets.py), models (models.py), experiments (experiments.py).

Other side responsabilities:
    - notifications (events.py)
    - changelog with notifications digest (changelog.py)
"""
import logging

# Format all loggers in the application to show the filename and linenumber
logging.basicConfig(
    format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
