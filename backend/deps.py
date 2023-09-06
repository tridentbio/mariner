"""Routines to install torch geometric dependenci activating the virtual environment first.

Adapted from https://github.com/BorgwardtLab/TOGL/blob/af2d0fb9e3262d1327fff512963e3d7745a6acae/deps.
"""

import os
import subprocess
import sys

# Script configuration:
platforms = ["cpu", "cu117", "cu118"]
default_platform = platforms[0]
default_torch_version = "2.0.1"

if not os.getenv("TORCH_VERSION"):
    print(
        f"TORCH_VERSION not set. Using default {default_torch_version}",
        file=sys.stderr,
    )
if not os.getenv("PLATFORM"):
    print(
        f"PLATFORM not set. Using default {default_platform}",
        file=sys.stderr,
    )
TORCH_VERSION = os.getenv("TORCH_VERSION", default_torch_version)
PLATFORM = os.getenv("PLATFORM", default_platform)


def make_torch_geometric_download_uri(platform: str, torch_version: str):
    """
    Creates the value for -f in pip install.

    Args:
        platform: The platform to install for.
        torch_version: The torch version to install for.
    """
    return f"https://data.pyg.org/whl/torch-{torch_version}+{platform}.html"


def make_torch_index_uri(platform: str):
    """
    Creates the value for --index-uri in pip install.

    Args:
        platform: The platform to install for.
    """
    return f"https://download.pytorch.org/whl/{platform}"


def install_deps(platform=PLATFORM, torch_version=TORCH_VERSION):
    pip_install_command = [
        "poetry",
        "run",
        "pip",
        "install",
        "--timeout",
        "120",
    ]
    deps = ["torch-scatter", "torch-sparse", "torch-geometric"]

    torch_install_exit = subprocess.call(
        pip_install_command
        + [
            f"torch=={torch_version}",
            "--index-url",
            make_torch_index_uri(platform),
        ]
    )
    if torch_install_exit != 0:
        print(
            f"Failed to install torch. Aborting installation of torch=={torch_version}",
            file=sys.stderr,
        )
        sys.exit(1)
    for pyg_dep in deps:
        pyg_install_exit = subprocess.call(
            pip_install_command
            + [
                pyg_dep,
                "-f",
                make_torch_geometric_download_uri(platform, torch_version),
            ]
        )

        if pyg_install_exit != 0:
            print(
                f"Failed to install torch geometric dependencies. Aborting installation of {','.join(deps)}"
            )
            sys.exit(1)


usage = f"""
python -m deps
    Environment variables:
        PLATFORM must be one of: {",".join(platforms)}. default {default_platform}
        TORCH_VERSION sets the torch version. default {default_torch_version}
"""

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(usage)
        sys.exit(0)
    install_deps()
