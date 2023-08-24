import { FormLabel } from '@mui/material';
import NotFound from 'components/atoms/NotFound';
import { ReactFlowProvider } from 'reactflow';
import ModelEditor from 'components/templates/ModelEditorV2';
import { modelsApi } from 'app/rtk/models';
import { ModelEditorContextProvider } from 'hooks/useModelEditor';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';
import {
  ModelCreate,
  SklearnModelSchema,
  TorchModelSpec,
} from '@app/rtk/generated/models';
import DataPreprocessingInput from '@components/organisms/ModelBuilder/DataPreprocessingInput';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import { FormProvider, useForm } from 'react-hook-form';
import { ModelBuilderContextProvider } from '@components/organisms/ModelBuilder/hooks/useModelBuilder';
import { useEffect } from 'react';
import SklearnModelInput from '@components/organisms/ModelBuilder/SklearnModelInput';

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
  const sklearnFormMethods = useForm<ModelCreate>();

  useEffect(() => {
    if (modelVersion) {
      sklearnFormMethods.reset({
        config: {
          dataset: modelVersion.config.dataset,
          spec: modelVersion.config.spec as SklearnModelSchema,
        },
      });
    }
  }, [modelVersion]);

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

  if (modelVersion.config.framework == 'sklearn') {
    return (
      <FormProvider {...sklearnFormMethods}>
        <ModelBuilderContextProvider editable={false} defaultExpanded={true}>
          <DataPreprocessingInput
            value={{
              featureColumns: modelVersion.config.dataset
                .featureColumns as SimpleColumnConfig[],
              targetColumns: modelVersion.config.dataset
                .targetColumns as SimpleColumnConfig[],
            }}
          />
          <SklearnModelInput />
        </ModelBuilderContextProvider>
      </FormProvider>
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
