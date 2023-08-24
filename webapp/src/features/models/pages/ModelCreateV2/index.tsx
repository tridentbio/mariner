import SklearnModelInput from '@components/organisms/ModelBuilder/SklearnModelInput';
import {
  preprocessingStepSchema,
  sklearnDatasetSchema,
  torchDatasetSchema,
} from '@components/organisms/ModelBuilder/formSchema';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import { yupResolver } from '@hookform/resolvers/yup';
import { LoadingButton } from '@mui/lab';
import { Button, Step, StepLabel, Stepper } from '@mui/material';
import { Box } from '@mui/system';
import { useNotifications } from 'app/notifications';
import * as modelsApi from 'app/rtk/generated/models';
import StackTrace from 'components/organisms/StackTrace';
import Content from 'components/templates/AppLayout/Content';
import TorchModelEditor from 'components/templates/TorchModelEditorV2';
import { TorchModelEditorContextProvider } from 'hooks/useTorchModelEditor';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';
import { MouseEvent, useEffect, useState } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { FieldPath, FormProvider, useForm } from 'react-hook-form';
import { useNavigate, useSearchParams } from 'react-router-dom';
import * as yup from 'yup';
import { DatasetConfigurationForm } from './DatasetConfigurationForm';
import ModelConfigForm from './ModelConfigForm';
import { ModelSetup } from './ModelSetup';

type ModelCreationStep = {
  title: string;
  stepId:
    | 'model-setup'
    | 'model-description'
    | 'dataset-configuration'
    | 'model-architecture';
  content: JSX.Element | null;
};

type FormFieldNames = FieldPath<modelsApi.ModelCreate>;

export const schema = yup.object({
  name: yup.string().required('Model name field is required'),
  config: yup.object({
    name: yup.string().required('Model version name is required'),
    framework: yup
      .string()
      .oneOf(['torch', 'sklearn'])
      .required('The framework is required'),
    dataset: yup.object().required().when('framework', {
      is: 'sklearn',
      then: sklearnDatasetSchema,
      otherwise: torchDatasetSchema,
    }),
    spec: yup.object().when('framework', {
      is: 'sklearn',
      then: yup
        .object({ model: preprocessingStepSchema.required() })
        .required('Sklearn model is required'),
      otherwise: yup.object({ layers: yup.array() }).required(),
    }),
  }),
});

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
          strategy: 'forwardArgs',
          featureColumns: [],
          featurizers: [],
          transforms: [],
          targetColumns: [],
        },
        spec: { layers: [] },
      },
    },
    resolver: yupResolver(schema),
  });

  const [createModel, { error, isLoading: creatingModel, data }] =
    modelsApi.useCreateModelMutation();
  const { control, getValues, setValue, watch } = methods;
  const config = watch('config');

  const selectedFramework: 'torch' | 'sklearn' = watch('config.framework');

  const onFrameworkChange = () => {
    if (selectedFramework == 'torch') {
      setValue('config', {
        ...config,
        dataset: {
          strategy: 'forwardArgs',
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
        },
        spec: { layers: [] },
      } as typeof config & { framework: 'torch' });
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

      setValue('config', {
        ...config,
        dataset: {
          strategy: 'pipeline',
          name: config.dataset.name,
          featureColumns: config.dataset.featureColumns.map(
            getSimpleColumnConfigTemplate
          ),
          targetColumns: config.dataset.targetColumns.map(
            getSimpleColumnConfigTemplate
          ),
        },
        spec: { model: null as modelsApi.SklearnModelSchema['model'] | null },
      } as typeof config & { framework: 'sklearn' });
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
      (errors) => {
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
    handleRegisteredModel();
  }, [registeredModel]);

  const handleRegisteredModel = async () => {
    if (!registeredModel) return;

    const result = await getExistingModel({
      modelId: parseInt(registeredModel),
    }).unwrap();

    methods.setValue('name', result.name);
    methods.setValue('modelDescription', result.description || '');
    methods.setValue('config.dataset.name', result?.dataset?.name || '');

    const lastVersion = result.versions.length
      ? result.versions[result.versions.length - 1]
      : null;

    methods.setValue(
      'config.framework',
      lastVersion ? lastVersion.config.framework : 'torch'
    );

    const modelFeatures = result.columns.filter(
      (col) => col.columnType === 'feature'
    );
    const modelTarget = result.columns.filter(
      (col) => col.columnType === 'target'
    );
    const featureColumns = result?.dataset?.columnsMetadata?.filter((meta) =>
      modelFeatures.map((feat) => feat.columnName).includes(meta.pattern)
    );

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
      content:
        selectedFramework == 'sklearn' ? <DatasetConfigurationForm /> : null,
    },
    {
      title: 'Model Architecture',
      stepId: 'model-architecture',
      content: (
        <Box sx={{ maxWidth: '100vw' }}>
          <div>
            {selectedFramework == 'torch' ? (
              <ReactFlowProvider>
                <TorchModelEditorContextProvider>
                  <TorchModelEditor
                    value={extendSpecWithTargetForwardArgs(
                      config as modelsApi.TorchModelSpec
                    )}
                    onChange={(value) => {
                      setValue(
                        'config',
                        value as modelsApi.ModelCreate['config']
                      );
                    }}
                  />
                </TorchModelEditorContextProvider>
              </ReactFlowProvider>
            ) : (
              <Box style={{ height: '45vh' }}>
                <SklearnModelInput />
              </Box>
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
                disabled={checkingModel || creatingModel}
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
              <LoadingButton
                loading={checkingModel || creatingModel}
                variant="contained"
                onClick={handleModelCreate}
              >
                <span>CREATE</span>
              </LoadingButton>
            )}
          </Box>
        </form>
      </FormProvider>
    </Content>
  );
};

export default ModelCreateV2;
