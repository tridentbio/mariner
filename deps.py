"""Routines to install torch geometric dependenci activating the virtual environment first.

Adapted from https://github.com/BorgwardtLab/TOGL/blob/af2d0fb9e3262d1327fff512963e3d7745a6acae/deps.
"""

import subprocess
import sys


def make_download_uri(platform: str, torch_version="1.13.0"):
    return f"https://data.pyg.org/whl/torch-{torch_version}+{platform}.html"


def install_deps(cuda):
    pip_install_command = ["poetry", "run", "pip", "install"]
    deps = ["torch-scatter", "torch-sparse", "torch-geometric"]
    for lib in deps:
        subprocess.call(pip_install_command + [lib, "-f", make_download_uri(cuda)])


def install_deps_cpu():
    install_deps("cpu")


def install_deps_cu101():
    install_deps("cu101")


def install_deps_cu102():
    install_deps("cu102")


def install_deps_cu110():
    install_deps("cu110")


envs = ["cpu", "cu101", "cu102", "cu110"]
usage = """
python -m deps [CUDA]
    CUDA must be one of: cpu,cu101,cu102,cu110. default: cpu
"""

if __name__ == "__main__":
    cuda = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if cuda and cuda not in envs:
        print(usage)
        sys.exit(1)
    install_deps(cuda)
