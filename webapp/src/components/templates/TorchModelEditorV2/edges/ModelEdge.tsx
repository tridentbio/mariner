import useTorchModelEditor from 'hooks/useTorchModelEditor';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import { getComponent } from 'model-compiler/src/implementation/modelSchemaQuery';
import { getBezierPath, EdgeProps } from 'reactflow';
import { EdgeButton, EdgeButtonContainer } from './styles';
import Suggestion from '@model-compiler/src/implementation/Suggestion';
import { isArray } from '@utils';
import { useMemo } from 'react';
interface ModelEdgeProps extends EdgeProps {
  editable?: boolean;
}

const foreignObjectSize = 60;

export const ModelEdge = ({ editable = true, ...props }: ModelEdgeProps) => {
  const {
    id,
    style,
    markerEnd,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    target,
    targetHandleId,
    sourceHandleId,
  } = props;

  const [path, labelX, labelY, offsetX, offsetY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const { suggestionsByEdge, editComponent, schema, options } =
    useTorchModelEditor();

  const edgeStrokeColor = useMemo(() => {
    const severityColorMap = new Map<
      Suggestion['severity'],
      { color: string; priority: number }
    >([
      ['ERROR', { color: 'red', priority: 1 }],
      ['WARNING', { color: '#ff7e14', priority: 2 }],
    ]);

    const suggestions = suggestionsByEdge[id];

    if (!isArray(suggestions)) return undefined;

    suggestions.sort(
      (a, b) =>
        severityColorMap.get(a.severity)!.priority -
        severityColorMap.get(b.severity)!.priority
    );

    for (const suggestion of suggestions) {
      if (severityColorMap.has(suggestion.severity))
        return severityColorMap.get(suggestion.severity)!.color;
    }

    return undefined;
  }, [id, suggestionsByEdge]);

  return (
    <>
      <path
        id={id}
        style={{
          ...(style || {}),
          stroke: edgeStrokeColor,
        }}
        className={'react-flow__edge-path'}
        d={path}
        markerEnd={markerEnd}
      />
      <foreignObject
        width={foreignObjectSize}
        height={foreignObjectSize}
        x={labelX - foreignObjectSize / 2}
        y={labelY - foreignObjectSize / 2}
        className="edgebutton-foreignobject"
        requiredExtensions="http://www.w3.org/1999/xhtml"
      >
        <EdgeButtonContainer editable={editable}>
          <EdgeButton
            onClick={(event) => {
              schema &&
                options &&
                editComponent(
                  {
                    data: makeComponentEdit({
                      component: getComponent(schema, target),
                      options,
                      removeConnection: {
                        targetNodeForwardArg: targetHandleId || '',
                        elementValue: sourceHandleId || undefined,
                      },
                    }),
                  },
                  schema
                );
            }}
          >
            ×
          </EdgeButton>
        </EdgeButtonContainer>
      </foreignObject>
    </>
  );
};

export default ModelEdge;
