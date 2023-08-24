import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import yaml

TEST_DIR = Path("tests") / "data"


def normalize_extension(ext: str):
    if ext in ["yaml", "yml"]:
        return "yaml"
    else:
        assert ext in ["json", "csv"], f"Unknown extension {ext}"
        return ext


@contextmanager
def load_test(filename: str, raw=False):
    # Gets filename extension
    ext = filename.split(".")[-1]
    path = TEST_DIR / normalize_extension(str(ext)) / filename
    file = open(path, "rb")
    if raw:
        yield file
    elif ext in ["yaml", "yml"]:
        yield yaml.safe_load(file)
    elif ext == "json":
        yield json.load(file)
    elif ext == "csv":
        yield pd.read_csv(file)
    else:
        raise ValueError('Unknown extension "{}"'.format(ext))
    file.close()
