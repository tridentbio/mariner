import { FormLabel } from '@mui/material';
import NotFound from 'components/atoms/NotFound';
import { ReactFlowProvider } from 'react-flow-renderer';
import ModelEditor from 'components/templates/ModelEditorV2';
import { modelsApi } from 'app/rtk/models';
import { ModelEditorContextProvider } from 'hooks/useModelEditor';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';

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
            <ModelEditor
              value={extendSpecWithTargetForwardArgs(modelVersion.config)}
              editable={false}
            />
          )}
        </ModelEditorContextProvider>
      </ReactFlowProvider>
    </>
  );
};

export default ModelVersionDetailsView;
