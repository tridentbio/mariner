import { FormLabel } from '@mui/material';
import NotFound from 'components/atoms/NotFound';
import { ReactFlowProvider } from 'reactflow';
import TorchModelEditor from 'components/templates/TorchModelEditorV2';
import { modelsApi } from 'app/rtk/models';
import { TorchModelEditorContextProvider } from 'hooks/useTorchModelEditor';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';
import { TorchModelSpec } from '@app/rtk/generated/models';

interface ModelVersionDetailsProps {
  modelName?: string;
  version?: string;
  modelVersionId: number;
  modelId: number;
}

const ModelVersionDetailsView = (props: ModelVersionDetailsProps) => {
  const model = modelsApi.useGetModelByIdQuery(props.modelId).data;
  const modelVersion = model?.versions?.find(
    (modelVersion) => modelVersion.id === props.modelVersionId
  );

  if (!model) {
    return <NotFound>Model {`"${props.modelName}"`} not found</NotFound>;
  } else if (!modelVersion) {
    return (
      <NotFound>
        Version {`"${props.version}"`} not found for model{' '}
        {`"${props.modelName}"`}
      </NotFound>
    );
  }

  return (
    <>
      <FormLabel>Model:</FormLabel>
      <ReactFlowProvider>
        <TorchModelEditorContextProvider>
          {modelVersion.config && (
            <TorchModelEditor // TODO: change to component that supports all frameworks (update TorchModelEditor to TorchTorchModelEditor)
              value={extendSpecWithTargetForwardArgs(
                modelVersion.config as TorchModelSpec
              )}
              editable={false}
            />
          )}
        </TorchModelEditorContextProvider>
      </ReactFlowProvider>
    </>
  );
};

export default ModelVersionDetailsView;
