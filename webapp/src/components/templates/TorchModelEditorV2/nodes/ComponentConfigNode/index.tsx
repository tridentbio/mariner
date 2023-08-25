import { DeleteOutlined, ExpandOutlined } from '@mui/icons-material';
import DocsModel from 'components/templates/TorchModelEditor/Components/DocsModel/DocsModel';
import BaseNode from 'components/templates/TorchModelEditorV2/nodes/BaseNode';
import useTorchModelEditor from 'hooks/useTorchModelEditor';
import { LayerFeaturizerType } from '@model-compiler/src/interfaces/torch-model-editor';
import { NodeProps } from 'reactflow';
import { substrAfterLast } from 'utils';
import CustomHandles from '../CustomHandles';
import NodeHeader from '../NodeHeader';
import ConstructorArgsInputs from './ConstructorArgsInputs';
import { memo } from 'react';

export interface ComponentConfigNodeProps
  extends NodeProps<LayerFeaturizerType> {
  editable?: boolean;
}

const ComponentConfigNode = ({
  editable = true,
  ...props
}: ComponentConfigNodeProps) => {
  const { options, deleteComponents, schema, toggleExpanded } =
    useTorchModelEditor();
  const option = (options || {})[props.data.type!];
  const { docs, docsLink } = option || {};

  return (
    <BaseNode
      key={props.id}
      title={substrAfterLast(props.data.type!, '.')}
      docs={docs}
      docsLink={docsLink}
      id={props.id}
      handlesElement={
        <CustomHandles nodeId={props.data.name} type={props.data.type} />
      }
      headerExtra={
        <NodeHeader
          options={[
            {
              icon: <DocsModel docs={docs} docsLink={docsLink} />,
              onClick: () => null,
              tip: 'Documentation',
            },
            {
              icon: (
                <ExpandOutlined fontSize="small" width="25px" height="25px" />
              ),
              onClick: () => toggleExpanded(props.id),
              tip: 'Expand',
            },
            {
              icon: (
                <DeleteOutlined fontSize="small" width="25px" height="25px" />
              ),
              onClick: () =>
                schema &&
                deleteComponents({
                  schema,
                  nodeId: props.data.name,
                }),
              tip: 'Delete',
            },
          ].map((a, idx) => ({ ...a, key: idx.toString() }))}
        />
      }
    >
      <ConstructorArgsInputs data={props.data} editable={editable} />
    </BaseNode>
  );
};

export default memo(ComponentConfigNode);
