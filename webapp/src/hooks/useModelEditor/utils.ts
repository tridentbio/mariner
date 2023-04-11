import { Edge, Node, Position } from 'react-flow-renderer';
import dagre from 'dagre';
import { DataTypeDomainKind } from 'app/types/domain/datasets';
import { DataType } from 'model-compiler/src/interfaces/model-editor';

/**
 * Apply an auto graph layout into ReactFlow's nodes.
 * Returns a copy of the nodes, repositioned
 */
export const positionNodes = <T>(
  nodes: Omit<Node<T>, 'position'>[],
  edges: Edge[],
  direction: 'TB' | 'LR' = 'TB',
  multiplier: number = 2
): Node<T>[] => {
  const isHorizontal = direction === 'LR';
  const dagreGraph = new dagre.graphlib.Graph();

  dagreGraph.setDefaultEdgeLabel(() => ({}));
  const nodeWidth = Number((260 + ((multiplier - 1) / 4) * 250).toFixed(0));
  const nodeHeight = Number((90 + (multiplier - 1) * 80).toFixed(0));

  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  let targetPosition: Position, sourcePosition: Position;

  if (isHorizontal) {
    targetPosition = Position.Left;
    sourcePosition = Position.Right;
  } else {
    targetPosition = Position.Top;
    sourcePosition = Position.Bottom;
  }

  const positionedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    const position = {
      // We are shifting the dagre node position (anchor=center center) to the
      // top left so it matches the React Flow node anchor point (top left).
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };

    return {
      ...node,
      targetPosition,
      sourcePosition,
      position,
    };
  });

  return positionedNodes;
};

export const fixDomainKindCasing = (dk: DataType['domainKind']): string => {
  if (dk === DataTypeDomainKind.Smiles) return dk.toUpperCase();
  else if (!dk) return '';
  else return dk.charAt(0).toUpperCase() + dk.slice(1).toLowerCase();
};
