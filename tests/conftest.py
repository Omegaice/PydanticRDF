import pytest
from rdflib import Graph, Namespace


@pytest.fixture
def graph(EX: Namespace) -> Graph:
    g = Graph()
    g.bind("ex", EX)
    return g


@pytest.fixture
def EX() -> Namespace:
    return Namespace("http://example.com/")
