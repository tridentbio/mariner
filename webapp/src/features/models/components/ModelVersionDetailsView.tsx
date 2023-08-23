import { FormLabel } from '@mui/material';
import NotFound from 'components/atoms/NotFound';
import { ReactFlowProvider } from 'reactflow';
import ModelEditor from 'components/templates/ModelEditorV2';
import { modelsApi } from 'app/rtk/models';
import { ModelEditorContextProvider } from 'hooks/useModelEditor';
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
        <ModelEditorContextProvider>
          {modelVersion.config && (
            <ModelEditor // TODO: change to component that supports all frameworks (update ModelEditor to TorchModelEditor)
              value={extendSpecWithTargetForwardArgs(
                modelVersion.config as TorchModelSpec
              )}
              editable={false}
            />
          )}
        </ModelEditorContextProvider>
      </ReactFlowProvider>
    </>
  );
};

export default ModelVersionDetailsView;
