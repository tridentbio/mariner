import { getEdgeCenter, ReactFlowInstance } from 'react-flow-renderer';

type NodeSchemaContext = {
  nodeId: string;
};
type EdgeSchemaContext = {
  edgeId: string;
};

type SchemaContext = NodeSchemaContext | EdgeSchemaContext;

export const SchemaContextTypeGuard = {
  isNodeSchema: (value: any): value is NodeSchemaContext => {
    return 'nodeId' in value;
  },
  isEdgeSchema: (value: any): value is EdgeSchemaContext => {
    return 'edgeId' in value;
  },
};

export const locateContext = (
  ctx: SchemaContext,
  value: Pick<ReactFlowInstance, 'getNode' | 'getEdge'>
): { x: number; y: number } | undefined => {
  if (SchemaContextTypeGuard.isNodeSchema(ctx)) {
    const node = value.getNode(ctx.nodeId);
    if (!node) return;
    return {
      x: node?.position.x + (node.width || 0) / 2,
      y: node?.position.y + (node.height || 0) / 2,
    };
  } else if (SchemaContextTypeGuard.isEdgeSchema(ctx)) {
    const edge = value.getEdge(ctx.edgeId);
    if (!edge) return;
    //const sourceX = edge.sourceNode?.position.x || 0;
    //const sourceY = edge.sourceNode?.position.y || 0;
    const targetX = edge.targetNode?.position.x || 0;
    const targetY = edge.targetNode?.position.y || 0;
    //const [edgeCenterX, edgeCenterY] = getEdgeCenter({
    //  sourceX,
    //  sourceY,
    //  targetX,
    //  targetY,
    //});
    return {
      x: targetX,
      y: targetY,
    };
  }
};

export default SchemaContext;
