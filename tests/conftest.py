import os
import random
from pathlib import Path

import pytest

from drybench.backends.mock import MockBackend
from drybench.items import generate_items, make_item
from drybench.reader import Reader, ReaderConfig


@pytest.fixture(scope="session")
def items():
    return generate_items(2, seed=0)


@pytest.fixture()
def item_k2():
    return make_item(random.Random(7), "t-k2", k=2, d=2)


@pytest.fixture()
def mock_reader():
    return Reader(MockBackend(), ReaderConfig())


@pytest.fixture(scope="session")
def assets_dir():
    d = os.environ.get("DRYBENCH_ASSETS")
    if not d or not (Path(d) / "gpt2-lm-head-10.onnx").exists():
        pytest.skip("set DRYBENCH_ASSETS to the directory with GPT-2 assets to run this test")
    return Path(d)
