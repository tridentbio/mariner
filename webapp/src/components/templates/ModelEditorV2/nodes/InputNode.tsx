import BaseNode from './BaseNode';
import { Input } from 'model-compiler/src/interfaces/model-editor';
import { NodeProps, Position } from 'react-flow-renderer';
import CustomHandle from './CustomHandle';

type InputNodeProps = NodeProps<Input>;

const InputNode = (props: InputNodeProps) => {
  return (
    <BaseNode
      id={props.id}
      title={props.data.name}
      handlesElement={
        <CustomHandle
          nodeId={props.data.name}
          total={1}
          order={0}
          type="source"
          position={Position.Bottom}
        />
      }
    />
  );
};

export default InputNode;
