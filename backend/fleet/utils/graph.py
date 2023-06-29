"""
Utility functions to operate on graphs.
"""
from typing import Any, Callable, Iterable, List, Tuple

import networkx as nx

from fleet.model_builder.utils import unwrap_dollar

Edge = Tuple[str, Any]


def make_graph(
    nodes: Iterable[Any],
    get_node_id: Callable[[Any], str],
    get_edges: Callable[[Any], List[Edge]],
) -> nx.DiGraph:
    """Utility function to create a graph.

    Creates a graph iterating the nodes, and using the other arguments to get
    the necessary information on that node. ``get_node_id`` is used to get a
    string that identifies that node among other items in ``nodes``. ``get_edges``
    is called on a node and must return a list of edges, described by the the
    destiny node id and the edge attributes.

    Args:
        nodes: A iterable object with the nodes.
        get_node_id: A function to get the node identifier from a node.
        get_edges: A function to get the edges from a node.

    Returns:
        The networkx graph.
    """
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(get_node_id(node))
    for node in nodes:
        node_id = get_node_id(node)
        edges = get_edges(node)
        for dst, attr in edges:
            graph.add_edge(dst, node_id, attr=attr)
    return graph


def make_graph_from_forward_args(nodes: Iterable[dict]) -> nx.DiGraph:
    """Creates a graph from objects with ``forward_args``.

    Args:
        nodes: Iterable with nodes.

    Returns:
        The graph.
    """

    def get_edges_from_forward_args(node: dict) -> List[Edge]:
        if "forwardArgs" not in node:
            return []
        forward_args = node["forwardArgs"]
        if forward_args is None:
            return []
        edges: List[Edge] = []
        if isinstance(forward_args, dict):
            for key, value in forward_args.items():
                if isinstance(value, str):
                    dst_and_attrs, _ = unwrap_dollar(value)
                    dst = dst_and_attrs.split(".", 1)[0]
                    edges.append((dst, key))
                elif isinstance(value, list):
                    for item in value:
                        dst_and_attrs, _ = unwrap_dollar(item)
                        dst = dst_and_attrs.split(".", 1)[0]
                        edges.append((dst, key))
        elif isinstance(forward_args, list):
            for item in forward_args:
                dst_and_attrs, _ = unwrap_dollar(item)
                dst = dst_and_attrs.split(".", 1)[0]
                edges.append((dst, None))
        return edges

    return make_graph(
        nodes, lambda node: node["name"], get_edges_from_forward_args
    )


def get_leaf_nodes(graph: nx.DiGraph) -> List[str]:
    """Gets the leaf nodes of a graph.

    Args:
        graph: The graph.

    Returns:
        A list with the leaf nodes.
    """
    return [node for node in graph.nodes if graph.out_degree(node) == 0]
