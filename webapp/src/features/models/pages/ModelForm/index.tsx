import { useLazyGetMyDatasetsQuery } from '@app/rtk/generated/datasets';
import { store } from '@app/store';
import SklearnModelInput from '@components/organisms/ModelBuilder/SklearnModelInput';
import {
  preprocessingStepSchema,
  sklearnDatasetSchema,
  torchDatasetSchema,
} from '@components/organisms/ModelBuilder/formSchema';
import { ModelBuilderContextProvider } from '@components/organisms/ModelBuilder/hooks/useModelBuilder';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import { yupResolver } from '@hookform/resolvers/yup';
import { LoadingButton } from '@mui/lab';
import {
  Button,
  CircularProgress,
  Step,
  StepLabel,
  Stepper,
} from '@mui/material';
import { Box } from '@mui/system';
import { useNotifications } from 'app/notifications';
import * as modelsApi from 'app/rtk/generated/models';
import StackTrace from 'components/organisms/StackTrace';
import Content from 'components/templates/AppLayout/Content';
import TorchModelEditor from 'components/templates/TorchModelEditorV2';
import { TorchModelEditorContextProvider } from 'hooks/useTorchModelEditor';
import { extendSpecWithTargetForwardArgs } from 'model-compiler/src/utils';
import { MouseEvent, useEffect, useMemo, useState } from 'react';
import { FieldPath, FormProvider, useForm } from 'react-hook-form';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { ReactFlowProvider } from 'reactflow';
import * as yup from 'yup';
import { DatasetConfigurationForm } from './DatasetConfigurationForm';
import ModelConfigForm from './ModelConfigForm';
import { ModelSetup } from './ModelSetup';

export interface ModelFormProps {
  mode?: 'creation' | 'fix';
}

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

const ModelForm = ({ mode = 'creation' }: ModelFormProps) => {
  const [currentMode, setCurrentMode] = useState<'creation' | 'fix'>(mode);
  const [activeStep, setActiveStep] = useState<number>(0);
  const [fetchDatasets] = useLazyGetMyDatasetsQuery();

  const [searchParams, setSearchParams] = useSearchParams();
  const routeParams = useParams();
  const navigate = useNavigate();

  const registeredModel = searchParams.get('registeredModel');

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

  const [createModel, { isLoading: creatingModel }] =
    modelsApi.useCreateModelMutation();

  const [updateModel, { isLoading: updatingModel }] =
    modelsApi.usePutModelVersionMutation();

  const isSubmittingModel = useMemo(
    () => creatingModel || updatingModel,
    [creatingModel, updatingModel]
  );

  const { control, getValues, setValue, watch, reset } = methods;
  const config = watch('config');
  const selectedFramework: 'torch' | 'sklearn' = watch('config.framework');

  const [modelVersionToFix, setModelVersionToFix] =
    useState<modelsApi.ModelVersion>();

  const modelId = useMemo(() => {
    return currentMode == 'creation' ? registeredModel : routeParams.modelId;
  }, [routeParams.modelId, registeredModel]);

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
    currentMode === 'creation' && onFrameworkChange();
  }, [selectedFramework]);

  const handleModelFix = async () => {
    const storedModels = store.getState().models.models as
      | modelsApi.Model[]
      | undefined;

    const foundModel = storedModels?.length
      ? storedModels.find(
          (model) => model.id === parseInt(routeParams.modelId as string)
        )
      : await getExistingModel({
          modelId: parseInt(routeParams.modelId as string),
        }).unwrap();

    if (foundModel) {
      const modelVersion = foundModel.versions.find(
        (version) =>
          version.id === parseInt(routeParams.modelVersionId as string)
      );

      if (!!modelVersion) {
        //? Fills dataset select input options
        await fetchDatasets({
          page: 0,
          perPage: 15,
          searchByName: foundModel.dataset?.name,
        });

        setModelVersionToFix(modelVersion);

        reset({
          name: foundModel.name,
          modelDescription: foundModel.description,
          modelVersionDescription: modelVersion.description,
          config: modelVersion.config,
        });

        //? Move to the last step
        onStepChange(steps.length - 1, 0);
      }
    }
  };

  useEffect(() => {
    currentMode == 'fix' && handleModelFix();
  }, [modelId, routeParams.modelVersionId]);

  const handleModelCreate = (event: MouseEvent) => {
    event.preventDefault();

    methods.handleSubmit(
      async (model) => {
        try {
          if (mode == 'creation') {
            const createdModel = await createModel({
              modelCreate: model,
            }).unwrap();

            navigate(`/models/${createdModel.id}`);
          } else {
            await updateModel({
              modelId: parseInt(modelId as string),
              modelVersionId: parseInt(routeParams.modelVersionId as string),
              modelVersionUpdate: {
                config: model.config,
              },
            }).unwrap();

            navigate(`/models/${modelId}`);
          }
        } catch (error) {
          notifyError('Unable to process, please adjust your model');
        }
      },
      (errors) => {
        notifyError('Error creating model');
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

  const handleClearRegisteredModel = () => {
    setSearchParams('', {
      replace: true,
      state: {},
    });

    setCurrentMode('creation');
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
      content: (
        <ModelConfigForm
          control={control}
          onClear={handleClearRegisteredModel}
          disabled={currentMode == 'fix'}
        />
      ),
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
          <StackTrace
            stackTrace={modelVersionToFix?.checkStackTrace}
            message={
              'An exception is raised during your model configuration for this dataset'
            }
          />
        </Box>
      ),
    },
  ];

  return (
    <Content>
      <FormProvider {...methods}>
        <ModelBuilderContextProvider>
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
            {currentMode == 'fix' && !modelVersionToFix ? (
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'row',
                  alignItems: 'center',
                  justifyContent: 'center',
                  my: 15,
                }}
              >
                <CircularProgress sx={{ mt: 3 }} size={50} />
              </Box>
            ) : (
              steps[activeStep].content
            )}
            <Box key="footer" sx={{ mt: 2, ml: 3 }}>
              {activeStep !== 0 && (
                <Button
                  onClick={handlePrevious}
                  variant="contained"
                  sx={{ mr: 3 }}
                  data-testid="previous"
                  disabled={isSubmittingModel}
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
                  loading={isSubmittingModel}
                  variant="contained"
                  onClick={handleModelCreate}
                >
                  <span>{currentMode == 'creation' ? 'CREATE' : 'UPDATE'}</span>
                </LoadingButton>
              )}
            </Box>
          </form>
        </ModelBuilderContextProvider>
      </FormProvider>
    </Content>
  );
};

export default ModelForm;
