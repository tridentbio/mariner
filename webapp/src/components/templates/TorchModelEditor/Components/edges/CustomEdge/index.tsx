import React from 'react';
import { EdgeProps, getBezierPath, Node, useNodes } from 'reactflow';
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

  const [path, labelX, labelY, offsetX, offsetY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <path
        id={id}
        style={style}
        className="react-flow__edge-path error"
        d={path}
        markerEnd={markerEnd}
      />
      {contentEditable && (
        <foreignObject
          width={foreignObjectSize}
          height={foreignObjectSize}
          x={labelX - foreignObjectSize / 2}
          y={labelY - foreignObjectSize / 2}
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
