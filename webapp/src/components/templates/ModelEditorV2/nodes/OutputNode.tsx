import BaseNode from './BaseNode';
import { Input } from 'model-compiler/src/interfaces/model-editor';
import { NodeProps, Position } from 'react-flow-renderer';
import CustomHandle from './CustomHandle';
import NodeHeader from './NodeHeader';
import { ExpandOutlined } from '@mui/icons-material';
import useModelEditor from 'hooks/useModelEditor';
import OutputNodeInputs from './ComponentConfigNode/OutputNodeInputs';

interface InputNodeProps extends NodeProps<Input> {
  editable?: boolean;
}

const OutputNode = ({ editable = true, ...props }: InputNodeProps) => {
  const { toggleExpanded } = useModelEditor();
  return (
    <BaseNode
      id={props.id}
      title={props.data.name}
      headerExtra={
        <NodeHeader
          options={[
            {
              icon: (
                <ExpandOutlined fontSize="small" width="25px" height="25px" />
              ),
              onClick: () => toggleExpanded(props.id),
              tip: 'Expand',
            },
          ].map((a, idx) => ({ ...a, key: idx.toString() }))}
        />
      }
      handlesElement={
        <CustomHandle
          nodeId={props.data.name}
          id=""
          total={1}
          order={0}
          type="target"
          position={Position.Top}
        />
      }
    >
      <OutputNodeInputs editable={editable} name={props.data.name} />
    </BaseNode>
  );
};

export default OutputNode;
