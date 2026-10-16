"""Runs mimir's tests and benchmarks on an NVIDIA GPU through Modal.

    pip install modal
    modal run ci/cuda/modal_test.py

copies this checkout into a container with the CUDA runtime libraries on
a T4 and runs ci/cuda/test.sh there. The Go module and build caches live
in a Modal volume, so a run compiles only what changed. The exit code is
the script's.
"""

import os
import subprocess
import sys

import modal

GO_VERSION = "1.27.0"  # as go.mod requires
CUDA_IMAGE = "nvidia/cuda:12.6.3-runtime-ubuntu24.04"
GPU = "T4"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("curl", "git")
    .run_commands(
        f"curl -fsSL https://go.dev/dl/go{GO_VERSION}.linux-amd64.tar.gz | tar -C /usr/local -xz"
    )
    .add_local_dir(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "/mimir",
        # MNIST downloads, and a log written into the checkout as the run goes.
        ignore=[".git", "data/mnist", "ckpt", "*.txt", "*.log"],
    )
)
app = modal.App("mimir-cuda", image=image)
cache = modal.Volume.from_name("mimir-go-cache", create_if_missing=True)


@app.function(gpu=GPU, timeout=3600, volumes={"/cache": cache})
def run(script: str) -> int:
    env = {
        **os.environ,
        "PATH": "/usr/local/go/bin:" + os.environ["PATH"],
        "GOMODCACHE": "/cache/mod",
        "GOCACHE": "/cache/build",
        "GOFLAGS": "-buildvcs=false",
        "MIMIR_CACHE_DIR": "/cache/mimir",
    }
    code = subprocess.call(["sh", script], cwd="/mimir", env=env)
    cache.commit()
    return code


@app.local_entrypoint()
def main(script: str = "ci/cuda/test.sh"):
    sys.exit(run.remote(script))
