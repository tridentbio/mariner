import { EPythonClasses } from 'app/types/domain/modelOptions';
import useTorchModelEditor from 'hooks/useTorchModelEditor';
import { LayerFeaturizerType } from '@model-compiler/src/interfaces/torch-model-editor';
import { Position } from 'reactflow';
import CustomHandle, { CustomHandleProps } from './CustomHandle';

export interface CustomHandlesProps {
  type: LayerFeaturizerType['type'];
  nodeId: string;
}

const CustomHandles = (props: CustomHandlesProps) => {
  const { options } = useTorchModelEditor();

  const getInputHandles = (): CustomHandleProps[] => {
    const { type, nodeId } = props;
    if (!type) return [];
    if (!options || !(type in options)) return [];
    const component = options[type].component;
    return Object.keys(component.forwardArgsSummary || {}).map(
      (key, index, A) => {
        return {
          nodeId,
          type: 'target',
          id: key,
          position: Position.Top,
          isConnectable: true,
          order: index,
          total: A.length,
        };
      }
    );
  };
  const getOutputHandles = (): CustomHandleProps[] => {
    const { type, nodeId } = props;
    if (!type) return [];
    if (!options || !(type in options)) return [];
    const outputType = options[type].outputType;
    if (outputType === EPythonClasses.TORCH_GEOMETRIC_DATA_REQUIRED) {
      return ['x', 'edge_index', 'edge_attr', 'batch'].map((key, index, A) => ({
        nodeId,
        id: key,
        type: 'source',
        position: Position.Bottom,
        isConnectable: true,
        order: index,
        total: A.length,
      }));
    } else {
      return [
        {
          nodeId,
          type: 'source',
          position: Position.Bottom,
          isConnectable: true,
          order: 0,
          total: 1,
        },
      ];
    }
  };
  const inputHandles: CustomHandleProps[] = getInputHandles();
  const outputHandles: CustomHandleProps[] = getOutputHandles();

  return (
    <>
      {inputHandles.concat(outputHandles).map((handleProps, index) => {
        return <CustomHandle {...handleProps} key={index} />;
      })}
    </>
  );
};

export default CustomHandles;
