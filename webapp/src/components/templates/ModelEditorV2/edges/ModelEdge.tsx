import useModelEditor from 'hooks/useModelEditor';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import { getComponent } from 'model-compiler/src/implementation/modelSchemaQuery';
import { getBezierPath, getEdgeCenter, EdgeProps } from 'react-flow-renderer';
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
  const { suggestionsByEdge, editComponent, schema, options } =
    useModelEditor();

  const getEdgeStrokeColor = useMemo(() => {
    const startMap = new Map<Suggestion['severity'], boolean>();

    const suggestionsSeverities = isArray(suggestionsByEdge[id])
      ? suggestionsByEdge[id].reduce(
          (acc, suggestion) => acc.set(suggestion.severity, true),
          startMap
        )
      : startMap;

    const colorStates: {
      [state in Suggestion['severity']]?: {
        getCondition: () => boolean;
        color: string;
      };
    } = {
      ERROR: {
        color: 'red',
        getCondition: () =>
          typeof suggestionsByEdge[id] == 'string' ||
          suggestionsSeverities.has('ERROR'),
      },
      WARNING: {
        color: '#ff7e14',
        getCondition: () => suggestionsSeverities.has('WARNING'),
      },
    };

    for (const state in colorStates) {
      let colorState = colorStates[state as keyof typeof colorStates];

      if (colorState?.getCondition())
        return colorStates[state as keyof typeof colorStates]?.color;
    }

    return undefined;
  }, [id, suggestionsByEdge]);

  return (
    <>
      <path
        id={id}
        style={{
          ...(style || {}),
          stroke: getEdgeStrokeColor,
        }}
        className={'react-flow__edge-path'}
        d={edgePath}
        markerEnd={markerEnd}
      />
      <foreignObject
        width={foreignObjectSize}
        height={foreignObjectSize}
        x={edgeCenterX - foreignObjectSize / 2}
        y={edgeCenterY - foreignObjectSize / 2}
        className="edgebutton-foreignobject"
        requiredExtensions="http://www.w3.org/1999/xhtml"
      >
        <EdgeButtonContainer editable={editable}>
          <EdgeButton
            onClick={(event) => {
              schema &&
                options &&
                editComponent({
                  schema,
                  data: makeComponentEdit({
                    component: getComponent(schema, target),
                    options,
                    removeConnection: {
                      targetNodeForwardArg: targetHandleId || '',
                      elementValue: sourceHandleId || undefined,
                    },
                  }),
                });
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
