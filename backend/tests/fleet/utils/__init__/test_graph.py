"""
Tests fleet.utils.graph
"""
from dataclasses import dataclass
from typing import Dict, List, Union

from fleet.utils.graph import make_graph, make_graph_from_forward_args


def test_make_graph():
    """Tests ``make_graph``."""
    nodes = [
        {"name": "a", "edges": [("b", 1), ("c", 2)]},
        {"name": "b", "edges": [("c", 3)]},
        {"name": "c", "edges": []},
    ]
    graph = make_graph(nodes, lambda node: node["name"], lambda node: node["edges"])
    assert set(graph.nodes) == {"a", "b", "c"}
    assert set(graph.edges) == {("a", "b"), ("a", "c"), ("b", "c")}
    assert graph["a"]["b"]["attr"] == 1
    assert graph["a"]["c"]["attr"] == 2
    assert graph["b"]["c"]["attr"] == 3


@dataclass
class TestAnyNode:
    """Example class showing the functions work with duck typing."""

    name: str
    forward_args: Dict[str, Union[str, List[str]]]


def test_make_graph_from_forward_args():
    """Tests ``make_graph_from_forward_args``."""
    nodes = [
        {"name": "a", "forwardArgs": {"inputa": "$b.output"}},
        {"name": "b", "forwardArgs": {"inputb": ["$c.a", "$d.b"]}},
        {"name": "c", "forwardArgs": {"inputc": "$c"}},
        {"name": "d", "forwardArgs": {"inputd": "$d"}},
    ]
    graph = make_graph_from_forward_args(nodes)
    assert set(graph.nodes) == {"a", "b", "c", "d"}
    assert set(graph.edges) == {
        ("a", "b"),
        ("b", "c"),
        ("b", "d"),
        ("c", "c"),
        ("d", "d"),
    }
    assert graph["a"]["b"]["attr"] == "inputa"
    assert graph["b"]["c"]["attr"] == "inputb"
    assert graph["b"]["d"]["attr"] == "inputb"
