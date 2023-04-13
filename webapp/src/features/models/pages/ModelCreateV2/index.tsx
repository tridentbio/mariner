import { FormEventHandler, MouseEvent, useEffect, useState } from 'react';
import { ModelEditorContextProvider } from '@hooks/useModelEditor';
import { FormProvider, useForm, useWatch } from 'react-hook-form';
import * as modelsApi from 'app/rtk/generated/models';
import ModelEditor from 'components/templates/ModelEditorV2';
import ModelConfigForm from './ModelConfigForm';
import DatasetConfigForm from './DatasetConfigForm';
import { ReactFlowProvider } from 'react-flow-renderer';
import Content from 'components/templates/AppLayout/Content';
import { useNotifications } from 'app/notifications';
import { useNavigate, useSearchParams } from 'react-router-dom';
import StackTrace from 'components/organisms/StackTrace';
import { ModelSchema } from '@model-compiler/src/interfaces/model-editor';
import { Box } from '@mui/system';
import { Stepper, Step, Container, Button, StepLabel } from '@mui/material';

const ModelCreateV2 = () => {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [checkModel, { isLoading: checkingModel, data: configCheckData }] =
    modelsApi.usePostModelCheckConfigMutation();
  const [searchParams, setSearchParams] = useSearchParams();
  const { notifyError } = useNotifications();
  const methods = useForm<modelsApi.ModelCreate & { config: ModelSchema }>({
    mode: 'all',
    defaultValues: {
      name: '',
      modelDescription: '',
      modelVersionDescription: '',
      config: {
        name: '',
        dataset: {
          featureColumns: [],
          targetColumns: [],
        },
        layers: [],
        featurizers: [],
      },
    },
  });
  const [createModel, { error, isLoading, data }] =
    modelsApi.useCreateModelMutation();
  const { control, getValues, setValue } = methods;
  const config = useWatch({ control, name: 'config' });
  const navigate = useNavigate();
  const handleModelCreate = (event: MouseEvent) => {
    event.preventDefault();
    methods.handleSubmit(
      async (modelCreate) => {
        const result = await checkModel({
          modelSchema: modelCreate.config,
        });
        if (!('error' in result) && !result.data.stackTrace)
          return createModel({
            modelCreate,
          }).then((result) => {
            if ('data' in result) {
              navigate(`/models/${result.data.id}`);
            }
          });
      },
      () => {
        notifyError('Error creating dataset');
      }
    )();
  };

  const onStepChange = (stepIndex: number, oldIndex: number) => {
    if (stepIndex < oldIndex) return setActiveStep(stepIndex);
    let error: string | undefined;
    if (oldIndex === 0) {
      // Validate model config step
      const modelName = getValues('name');
      const modelVersionName = getValues('config.name');
      if (!modelName) error = 'Missing model name';
      else if (!modelVersionName) error = 'Missing model version name';
    } else if (oldIndex === 1) {
      // Validate Dataset config step
      const dataset = getValues('config.dataset');
      if (!dataset?.name) error = 'Missing dataset name';
      else if (!dataset.targetColumns?.length)
        error = 'Missing dataset target column selection';
      else if (!dataset.featureColumns?.length)
        error = 'Missing dataset feature columns selection';
    } else if (oldIndex === 2) {
      // Validate model architecture step
    }
    if (!error) {
      setActiveStep(stepIndex);
    } else {
      notifyError(error);
    }
  };

  const [getExistingModel, { data: existingModel, isLoading: fetchingModel }] =
    modelsApi.useLazyGetModelQuery();
  const registeredModel = searchParams.get('registeredModel');
  useEffect(() => {
    const go = async () => {
      if (!registeredModel) return;
      const result = await getExistingModel({
        modelId: parseInt(registeredModel),
      }).unwrap();
      methods.setValue('name', result.name);
      methods.setValue('modelDescription', result.description || '');
      methods.setValue('config.dataset.name', result?.dataset?.name || '');
      const modelFeatures = result.columns.filter(
        (col) => col.columnType === 'feature'
      );
      const modelTarget = result.columns.filter(
        (col) => col.columnType === 'target'
      );
      const featureColumns = result?.dataset?.columnsMetadata?.filter((meta) =>
        modelFeatures.map((feat) => feat.columnName).includes(meta.pattern)
      );
      // const targetColumns = result?.dataset?.columnsMetadata?.find(
      //   (meta) => modelTarget?.columnName === meta.pattern
      // );
      const targetColumns = result?.dataset?.columnsMetadata?.filter((meta) =>
        modelTarget.map((target) => target.columnName).includes(meta.pattern)
      );
      const makeColumnConfigFromDescription = (
        description: modelsApi.ColumnsDescription
      ): modelsApi.ColumnConfig => {
        return {
          name: description.pattern,
          dataType: description.dataType,
        };
      };
      if (featureColumns)
        methods.setValue(
          'config.dataset.featureColumns',
          featureColumns.map(makeColumnConfigFromDescription)
        );

      if (targetColumns)
        methods.setValue(
          'config.dataset.targetColumns',
          targetColumns.map(makeColumnConfigFromDescription)
        );
    };
    go();
  }, [registeredModel]);

  const handlePrevious = () => {
    const newStep = activeStep - 1;
    setActiveStep(newStep);
  };
  const handleNext = () => {
    const newStep = activeStep + 1;
    setActiveStep(newStep);
  };

  const steps = [
    {
      title: 'Model Description',
      content: <ModelConfigForm control={control} />,
    },
    {
      title: 'Features and Target',
      content: <DatasetConfigForm control={control} />,
    },
    {
      title: 'Model Architecture',
      content: (
        <Box sx={{ maxWidth: '100vw' }}>
          <div>
            <ModelEditor
              value={config}
              onChange={(value) => {
                setValue('config', value);
              }}
            />
          </div>
          {configCheckData?.stackTrace && (
            <StackTrace
              stackTrace={configCheckData?.stackTrace}
              message={
                'An exception is raised during your model configuration for this dataset'
              }
            />
          )}
        </Box>
      ),
    },
  ];

  return (
    <ReactFlowProvider>
      <ModelEditorContextProvider>
        <Content>
          <FormProvider {...methods}>
            <form style={{ position: 'relative', overflowX: 'hidden' }}>
              <Stepper
                orientation={'horizontal'}
                activeStep={activeStep}
                sx={{ mb: 5, mt: 2 }}
              >
                {steps.map(({ title }) => (
                  <Step key={title}>
                    <StepLabel>{title}</StepLabel>
                  </Step>
                ))}
              </Stepper>
              {steps[activeStep].content}
              <Box key="footer" sx={{ mt: 2, ml: 3 }}>
                {activeStep !== 0 && (
                  <Button
                    onClick={handlePrevious}
                    variant="contained"
                    sx={{ mr: 3 }}
                    data-testid="previous"
                  >
                    PREVIOUS
                  </Button>
                )}
                {activeStep !== steps.length - 1 && (
                  <Button
                    data-testid="next"
                    onClick={handleNext}
                    variant="contained"
                  >
                    NEXT
                  </Button>
                )}
                {activeStep === steps.length - 1 && (
                  <Button variant="contained" onClick={handleModelCreate}>
                    CREATE
                  </Button>
                )}
              </Box>
            </form>
          </FormProvider>
        </Content>
      </ModelEditorContextProvider>
    </ReactFlowProvider>
  );
};

export default ModelCreateV2;
