"""
Tests fleet.utils.graph
"""

from fleet.utils.graph import (
    get_leaf_nodes,
    make_graph,
    make_graph_from_forward_args,
)


def test_make_graph():
    """Tests ``make_graph``."""
    nodes = [
        {"name": "a", "edges": [("b", 1), ("c", 2)]},
        {"name": "b", "edges": [("c", 3)]},
        {"name": "c", "edges": []},
    ]
    graph = make_graph(
        nodes, lambda node: node["name"], lambda node: node["edges"]
    )
    assert set(graph.nodes) == {"a", "b", "c"}
    assert set(graph.edges) == {("b", "a"), ("c", "a"), ("c", "b")}
    assert graph["b"]["a"]["attr"] == 1
    assert graph["c"]["a"]["attr"] == 2
    assert graph["c"]["b"]["attr"] == 3


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
        ("b", "a"),
        ("c", "b"),
        ("d", "b"),
        ("c", "c"),
        ("d", "d"),
    }
    assert graph["b"]["a"]["attr"] == "inputa"
    assert graph["c"]["b"]["attr"] == "inputb"
    assert graph["d"]["b"]["attr"] == "inputb"


def test_get_leaf_nodes():
    """Tests ``get_leaf_nodes``."""
    nodes = [
        {"name": "a", "forwardArgs": {"inputa": "$b.output"}},
        {"name": "b", "forwardArgs": {"inputb": ["$c.a", "$d.b"]}},
        {"name": "c", "forwardArgs": {"inputc": "$c"}},
        {"name": "d", "forwardArgs": {"inputd": "$d"}},
    ]
    graph = make_graph_from_forward_args(nodes)
    leaf_nodes = get_leaf_nodes(graph)
    assert set(leaf_nodes) == {"a"}
