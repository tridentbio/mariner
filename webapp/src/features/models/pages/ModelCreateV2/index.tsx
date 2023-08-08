import { Button, Step, StepLabel, Stepper } from '@mui/material';
import { Box } from '@mui/system';
import { useNotifications } from 'app/notifications';
import * as modelsApi from 'app/rtk/generated/models';
import StackTrace from 'components/organisms/StackTrace';
import Content from 'components/templates/AppLayout/Content';
import ModelEditor from 'components/templates/ModelEditorV2';
import { ModelEditorContextProvider } from 'hooks/useModelEditor';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';
import { MouseEvent, useEffect, useState } from 'react';
import { ReactFlowProvider } from 'react-flow-renderer';
import { FieldPath, FormProvider, useForm } from 'react-hook-form';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { DatasetConfigurationForm } from './DatasetConfigurationForm';
import ModelConfigForm from './ModelConfigForm';
import { ModelSetup } from './ModelSetup';
import * as yup from 'yup';
import { yupResolver } from '@hookform/resolvers/yup';
import { simpleColumnSchema } from '@components/organisms/ModelBuilder/formSchema';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';

type ModelCreationStep = {
  title: string;
  stepId:
    | 'model-setup'
    | 'model-description'
    | 'dataset-configuration'
    | 'model-architecture';
  content: JSX.Element;
};

export const modelCreateSchema = yup.object({
  name: yup.string().required('Model name field is required'),
  config: yup.object({
    name: yup.string().required('Model version name is required'),
    framework: yup
      .string()
      .oneOf(['torch', 'sklearn'])
      .required('The framework is required'),
    dataset: yup.object({
      name: yup.string().required('Dataset is required'),
      featureColumns: yup
        .array()
        .required('The feature columns are required')
        .min(1, 'The feature columns must not be empty')
        .of(simpleColumnSchema),
      targetColumns: yup
        .array()
        .required()
        .min(1, 'The target columns must not be empty')
        .of(simpleColumnSchema),
    }),
  }),
});

type FormFieldNames = FieldPath<modelsApi.ModelCreate>;

const ModelCreateV2 = () => {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [checkModel, { isLoading: checkingModel, data: configCheckData }] =
    modelsApi.usePostModelCheckConfigMutation();
  const [searchParams, setSearchParams] = useSearchParams();
  const { notifyError } = useNotifications();

  const methods = useForm<modelsApi.ModelCreate>({
    mode: 'all',
    reValidateMode: 'onBlur',
    defaultValues: {
      name: '',
      modelDescription: '',
      modelVersionDescription: '',
      config: {
        name: '',
        framework: 'torch',
        dataset: {
          featureColumns: [],
          featurizers: [],
          transforms: [],
          targetColumns: [],
        },
        spec: { layers: [] },
      },
    },
    resolver: yupResolver(modelCreateSchema),
  });

  const [createModel, { error, isLoading, data }] =
    modelsApi.useCreateModelMutation();
  const { control, getValues, setValue, watch, unregister } = methods;
  const config = watch('config');

  const selectedFramework: 'torch' | 'sklearn' = watch('config.framework');

  const onFrameworkChange = () => {
    if (selectedFramework == 'torch') {
      setValue('config.dataset', {
        name: config.dataset.name,
        featureColumns: config.dataset.featureColumns.map((column) => ({
          name: column.name,
          dataType: column.dataType,
        })),
        targetColumns: config.dataset.targetColumns.map((column) => ({
          name: column.name,
          dataType: column.dataType,
          outModule: '',
        })),
        featurizers: [],
        transforms: [],
      } as modelsApi.TorchDatasetConfig);
    } else {
      const getSimpleColumnConfigTemplate = (
        column: modelsApi.ColumnConfig | SimpleColumnConfig
      ) =>
        ({
          name: column.name,
          dataType: column.dataType,
          featurizers: [],
          transforms: [],
        } as SimpleColumnConfig);

      setValue('config.dataset', {
        name: config.dataset.name,
        featureColumns: config.dataset.featureColumns.map(
          getSimpleColumnConfigTemplate
        ),
        targetColumns: config.dataset.targetColumns.map(
          getSimpleColumnConfigTemplate
        ),
      } as modelsApi.DatasetConfig);
    }
  };

  useEffect(() => {
    onFrameworkChange();
  }, [selectedFramework]);

  const navigate = useNavigate();
  const handleModelCreate = (event: MouseEvent) => {
    event.preventDefault();
    methods.handleSubmit(
      async (modelCreate) => {
        if (modelCreate.config.framework === 'torch') {
          const result = await checkModel({
            trainingCheckRequest: {
              modelSpec: modelCreate.config,
            },
          });
          if ('error' in result || result.data.stackTrace)
            return notifyError('Error creating dataset');
        }
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

  const revalidateErrorsOnFields = (field: FormFieldNames[]) =>
    field.forEach((field) => methods.trigger(field));

  const onStepChange = (stepIndex: number, oldIndex: number) => {
    const direction = stepIndex < oldIndex ? 'backward' : 'forward';

    const previousStepPageId = steps[oldIndex].stepId;
    const config = getValues('config');

    try {
      if (direction == 'forward') {
        switch (previousStepPageId) {
          case 'model-description':
            revalidateErrorsOnFields(['name', 'config.name']);

            const modelName = getValues('name');

            if (!modelName) throw 'Missing model name';
            if (!config.name) throw 'Missing model version name';
            break;
          case 'model-setup':
            revalidateErrorsOnFields(['config.dataset']);

            if (!config.dataset?.name) throw 'Missing dataset name';
            if (!config.dataset.targetColumns?.length)
              throw 'Missing dataset target column selection';
            if (!config.dataset.featureColumns?.length)
              throw 'Missing dataset feature columns selection';

            if (config.framework == 'torch')
              return moveToStepById('model-architecture');
            break;
          case 'dataset-configuration':
            revalidateErrorsOnFields(['config.dataset']);

            if (methods.getFieldState('config.dataset.targetColumns').invalid)
              throw 'Invalid target columns';
            if (methods.getFieldState('config.dataset.featureColumns').invalid)
              throw 'Invalid feature columns';
            break;
        }
      } else {
        switch (previousStepPageId) {
          case 'model-architecture':
            if (config.framework == 'torch')
              return moveToStepById('model-setup');
        }
      }

      setActiveStep(stepIndex);
    } catch (error) {
      notifyError(error as string);
    }
  };

  const moveToStepById = (stepId: ModelCreationStep['stepId']) => {
    const stepIndex = steps.findIndex((step) => step.stepId === stepId);
    if (stepIndex < 0) return;
    setActiveStep(stepIndex);
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
      const makeTargetColumnConfigFromDescription = (
        description: modelsApi.ColumnsDescription
      ): modelsApi.TargetTorchColumnConfig => {
        return {
          name: description.pattern,
          dataType: description.dataType,
          outModule: '',
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
          targetColumns.map(makeTargetColumnConfigFromDescription)
        );
    };
    go();
  }, [registeredModel]);

  const handlePrevious = () => {
    const newStep = activeStep - 1;
    onStepChange(newStep, activeStep);
  };
  const handleNext = () => {
    const newStep = activeStep + 1;
    onStepChange(newStep, activeStep);
  };

  const steps: ModelCreationStep[] = [
    {
      title: 'Model Description',
      stepId: 'model-description',
      content: <ModelConfigForm control={control} />,
    },
    {
      title: 'Model Setup',
      stepId: 'model-setup',
      content: <ModelSetup control={control} />,
    },
    {
      title: 'Dataset Configuration',
      stepId: 'dataset-configuration',
      content: <DatasetConfigurationForm />,
    },
    {
      title: 'Model Architecture',
      stepId: 'model-architecture',
      content: (
        <Box sx={{ maxWidth: '100vw' }}>
          <div>
            {selectedFramework == 'torch' ? (
              <ModelEditor
                value={extendSpecWithTargetForwardArgs(
                  config as modelsApi.TorchModelSpec
                )}
                onChange={(value) => {
                  setValue('config', value as modelsApi.ModelCreate['config']);
                }}
              />
            ) : (
              <div>empty</div>
            )}
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
