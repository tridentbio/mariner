import BaseNode from './BaseNode';
import { Input } from '@model-compiler/src/interfaces/torch-model-editor';
import { NodeProps, Position } from 'reactflow';
import CustomHandle from './CustomHandle';
import { memo } from 'react';

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

export default memo(InputNode);
