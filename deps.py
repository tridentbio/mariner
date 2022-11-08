# https://github.com/BorgwardtLab/TOGL/blob/af2d0fb9e3262d1327fff512963e3d7745a6acae/deps.
"""Routines to install torch geometric dependencies."""
import subprocess


def make_download_uri(platform: str, torch_version="1.12.1"):
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
