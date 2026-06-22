from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module

from .fakes import FakeGraph


@pytest.fixture
def client():
    # TestClient without `with` does not run lifespan, so we wire a fake graph
    # and skip the Chroma index / Groq dependencies.
    main_module._request_times.clear()
    main_module.app.state.graph = FakeGraph()
    return TestClient(main_module.app)


@pytest.fixture
def failing_client():
    main_module._request_times.clear()
    main_module.app.state.graph = FakeGraph(fail=True)
    return TestClient(main_module.app)
