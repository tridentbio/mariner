from subprocess import CalledProcessError, check_output

import pytest
import time
import xml.etree.ElementTree as ET

from model_builder import generate


def test_generate_bundle():
    python_code = generate.generate_bundlev2()
    assert isinstance(python_code, str)
    try:
        check_output(["python"], input=python_code, text=True)
    except CalledProcessError:
        pytest.fail("Failed to exectute generated python bundle`")


def test_sphinxfy():
    html_str = generate.sphinxfy("torch.nn.Linear")
    assert len(html_str) > 0, "sphinxfy did not produce output"
    root = ET.fromstring(html_str)
    elements = root.findall(r".//span[@class='math notranslate nohighlight']")
    assert len(elements) > 0, "failed to produce renderable math in html output"


@pytest.mark.benchmark
def test_benchmark_sphinxfy():
    import numpy as np

    tasks = generate.layers * 500
    times = np.array([0] * len(tasks))
    size_cache = 0
    for idx, layer in enumerate(tasks):
        t0 = time.time_ns()
        result = generate.sphinxfy(layer.name)
        times[idx] = time.time_ns() - t0
        if idx < len(generate.layers):  # only unrepeated args
            size_cache += len(result)

    total = times.sum()
    avg = times.mean()
    mx = times.max()
    mn = times.min()
    print(
        f"""
Benchmarking the model_builder.generate.sphinxfy
## TIME
    Number of docstrings transformed into HTML: %d
    total %0.4f seconds
    mean %0.4f seconds
    max %0.4f seconds
    min %0.4f seconds

## SPACE
    Number of classes cached: %d
    total %d bytes
    """
        % (
            len(tasks),
            total / 1e9,
            avg / 1e9,
            mx / 1e9,
            mn / 1e9,
            len(generate.layers),
            size_cache,
        ),
    )
