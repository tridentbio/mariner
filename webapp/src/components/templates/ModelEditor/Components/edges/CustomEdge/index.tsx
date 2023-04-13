import React from 'react';
import {
  EdgeProps,
  getBezierPath,
  getEdgeCenter,
  Node,
  useNodes,
} from 'react-flow-renderer';
import { EdgeButton, EdgeButtonContainer } from './styles';

const foreignObjectSize = 60;

interface CustomEdgeProps extends EdgeProps<any> {
  onRemoveEdgeClick: (
    evt: React.MouseEvent,
    id: string,
    nodes: Node<any>[]
  ) => void;
  contentEditable?: boolean;
}

export const CustomEdge: React.FC<CustomEdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = { stroke: 'GrayText', strokeWidth: '2' },
  markerEnd,
  onRemoveEdgeClick = () => {},
  contentEditable = true,
}) => {
  const nodes = useNodes();
  const edgePath = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  const [edgeCenterX, edgeCenterY] = getEdgeCenter({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });
  return (
    <>
      <path
        id={id}
        style={style}
        className="react-flow__edge-path error"
        d={edgePath}
        markerEnd={markerEnd}
      />
      {contentEditable && (
        <foreignObject
          width={foreignObjectSize}
          height={foreignObjectSize}
          x={edgeCenterX - foreignObjectSize / 2}
          y={edgeCenterY - foreignObjectSize / 2}
          className="edgebutton-foreignobject"
          requiredExtensions="http://www.w3.org/1999/xhtml"
        >
          <EdgeButtonContainer>
            <EdgeButton
              onClick={(event) => onRemoveEdgeClick(event, id, nodes)}
            >
              ×
            </EdgeButton>
          </EdgeButtonContainer>
        </foreignObject>
      )}
    </>
  );
};
