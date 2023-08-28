import { ModelCreate, TorchModelSpec } from '@app/rtk/generated/models';
import { Text } from '@components/molecules/Text';
import DataPreprocessingInput from '@components/organisms/ModelBuilder/DataPreprocessingInput';
import SklearnModelInput from '@components/organisms/ModelBuilder/SklearnModelInput';
import { ModelBuilderContextProvider } from '@components/organisms/ModelBuilder/hooks/useModelBuilder';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import {
  Container,
  FormLabel,
  Step,
  StepContent,
  StepLabel,
  Stepper,
} from '@mui/material';
import { modelsApi } from 'app/rtk/models';
import NotFound from 'components/atoms/NotFound';
import TorchModelEditor from 'components/templates/TorchModelEditorV2';
import { TorchModelEditorContextProvider } from 'hooks/useTorchModelEditor';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';
import { useEffect } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import { ReactFlowProvider } from 'reactflow';

interface ModelVersionDetailsProps {
  modelName?: string;
  version?: string;
  modelVersionId: number;
  modelId: number;
}

const ModelVersionDetailsView = (props: ModelVersionDetailsProps) => {
  const model = modelsApi.useGetModelByIdQuery(props.modelId).data;
  const sklearnFormMethods = useForm<ModelCreate>();

  const modelVersion = model?.versions?.find(
    (modelVersion) => modelVersion.id === props.modelVersionId
  );

  useEffect(() => {
    if (modelVersion?.config) {
      sklearnFormMethods.setValue(
        'config',
        {
          dataset: modelVersion?.config.dataset,
          spec: modelVersion?.config.spec,
          framework: 'sklearn',
          name: '',
        } as ModelCreate['config'] & { framework: 'sklearn' },
        { shouldValidate: true }
      );
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

  const isSklearnFormFilled =
    Object.keys(sklearnFormMethods.watch() || {}).length > 0;

  if (modelVersion.config.framework == 'sklearn') {
    if (!isSklearnFormFilled) return null;

    return (
      <FormProvider {...sklearnFormMethods}>
        <ModelBuilderContextProvider editable={false} defaultExpanded={true}>
          <Container>
            <Stepper orientation="vertical">
              <Step active>
                <StepContent>
                  <StepLabel>
                    <Text variant="subtitle1">Feature columns</Text>
                  </StepLabel>
                  <DataPreprocessingInput
                    value={
                      modelVersion.config.dataset
                        .featureColumns as SimpleColumnConfig[]
                    }
                    type="featureColumns"
                  />
                </StepContent>
              </Step>
              <Step active>
                <StepContent>
                  <StepLabel>
                    <Text variant="subtitle1">Target columns</Text>
                  </StepLabel>
                  <DataPreprocessingInput
                    value={
                      modelVersion.config.dataset
                        .targetColumns as SimpleColumnConfig[]
                    }
                    type="targetColumns"
                  />
                </StepContent>
              </Step>
              <Step active>
                <StepContent>
                  <StepLabel>
                    <Text variant="subtitle1">Model</Text>
                  </StepLabel>
                  <SklearnModelInput />
                </StepContent>
              </Step>
            </Stepper>
          </Container>
        </ModelBuilderContextProvider>
      </FormProvider>
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
