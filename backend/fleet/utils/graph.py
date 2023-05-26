"""
Utility functions to operate on graphs.
"""
from typing import Any, Callable, Dict, Iterable, List, Protocol, Tuple, Union

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
            graph.add_edge(node_id, dst, attr=attr)
    return graph


class AnyNode(Protocol):
    """
    A generic protocol for nodes used in the application.

    Attributes:
        name: the node identifier.
        forwardArgs: dictionary with edges data. Keys in this dictionary are
            function parameters (e.g. ``torch.nn.Linear``'s ``forward(input=...)``,
            ``input``is a function parameter that would be in the ``forward_arg``'s key of
            the layer's node representation). Values are strings or list of strings
            describing the value to be applied on that parameter (.e.g.
            ``"$foo.a.b"`` represents the parameter value is gotten from the ``foo``'s
            output, then accessing it's ``a`` and ``b`` attributes respectively.
    """

    name: str
    forward_args: Dict[str, Union[str, List[str]]]


def make_graph_from_forward_args(nodes: Iterable[AnyNode]) -> nx.DiGraph:
    """Creates a graph from objects with ``forward_args``.

    Args:
        nodes: Iterable with nodes.

    Returns:
        The graph.
    """

    def get_edges_from_forward_args(node: AnyNode) -> List[Edge]:
        edges: List[Edge] = []
        for key, value in node.forward_args.items():
            if isinstance(value, str):
                dst_and_attrs, _ = unwrap_dollar(value)
                dst = dst_and_attrs.split(".", 1)[0]
                edges.append((dst, key))
            elif isinstance(value, list):
                for item in value:
                    dst_and_attrs, _ = unwrap_dollar(item)
                    dst = dst_and_attrs.split(".", 1)[0]
                    edges.append((dst, key))
        return edges

    return make_graph(nodes, lambda node: node.name, get_edges_from_forward_args)
