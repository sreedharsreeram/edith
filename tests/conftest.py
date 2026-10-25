import pytest
from fastapi.testclient import TestClient
from helpers import FakeTranscriber, ScriptedModel, Step

from server.app import create_app


@pytest.fixture
def stt() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel(*[Step(text=["A ", "red ", "apple."]) for _ in range(10)])


@pytest.fixture
def make_client(stt, tmp_path):
    """Build a server around a given model and tools (default: no tools)."""
    clients = []

    def make(model, tools=None, token=""):
        client = TestClient(create_app(transcribe=stt, model=model, tools=tools or [], token=token, data_dir=tmp_path))
        clients.append(client.__enter__())
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client, model):
    return make_client(model)
