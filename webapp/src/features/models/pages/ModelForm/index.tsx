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
import TorchModelEditor, {
  ModelEditorElementsCount,
} from 'components/templates/TorchModelEditorV2';
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
import { NonUndefined } from '@utils';

export interface ModelFormProps {
  mode?: 'creation' | 'fix' | 'duplication';
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
  const [currentMode, setCurrentMode] =
    useState<NonUndefined<ModelFormProps['mode']>>(mode);
  const [activeStep, setActiveStep] = useState<number>(0);

  const [searchParams, setSearchParams] = useSearchParams();
  const routeParams = useParams();
  const navigate = useNavigate();

  const registeredModel = searchParams.get('registeredModel');
  const duplicationModelReference = searchParams.get(
    'duplicateFromModelVersion'
  );

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

  const { control, getValues, setValue, watch, reset } = methods;
  const config = watch('config');
  const selectedFramework: 'torch' | 'sklearn' = watch('config.framework');

  const [fetchDatasets] = useLazyGetMyDatasetsQuery();

  const [createModel, { isLoading: creatingModel }] =
    modelsApi.useCreateModelMutation();

  const [updateModel, { isLoading: updatingModel }] =
    modelsApi.usePutModelVersionMutation();

  const [getExistingModel] = modelsApi.useLazyGetModelQuery();

  const [getNameSuggestion] = modelsApi.useLazyGetModelNameSuggestionQuery();

  const isSubmittingModel = useMemo(
    () => creatingModel || updatingModel,
    [creatingModel, updatingModel]
  );

  const [modelVersionReference, setModelVersionReference] =
    useState<modelsApi.ModelVersion>();

  const [modelEditorNodesCount, setModelEditorNodesCount] =
    useState<ModelEditorElementsCount>({
      components: 0,
      templates: 0,
    });

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
          targetColumns: (
            config.dataset.targetColumns as modelsApi.TargetTorchColumnConfig[]
          ).map((column) => ({
            name: column.name,
            dataType: column.dataType,
            outModule: column.outModule,
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

  useEffect(() => {
    if (
      currentMode == 'fix' &&
      modelVersionReference &&
      modelVersionReference.checkStatus !== 'FAILED'
    ) {
      notifyError('Unable to modify non failed model versions');

      navigate(`/models/${routeParams.modelId}`);
    }
  }, [currentMode, modelVersionReference]);

  useEffect(() => {
    if (duplicationModelReference) setCurrentMode('duplication');
  }, [duplicationModelReference]);

  const fillFormByModel = async (
    modelId: modelsApi.Model['id'],
    modelVersionId: modelsApi.ModelVersion['id']
  ) => {
    const storedModels = store.getState().models.models as
      | modelsApi.Model[]
      | undefined;

    const foundModel = storedModels?.length
      ? storedModels.find((model) => model.id === modelId)
      : await getExistingModel({
          modelId: modelId,
        }).unwrap();

    if (foundModel) {
      const modelVersion = foundModel.versions.find(
        (version) => version.id === modelVersionId
      );

      if (!!modelVersion) {
        //? Fills dataset select input options
        await fetchDatasets({
          page: 0,
          perPage: 15,
          searchByName: foundModel.dataset?.name,
        });

        setModelVersionReference(modelVersion);

        reset({
          name: foundModel.name,
          modelDescription: foundModel.description,
          modelVersionDescription: modelVersion.description,
          config: modelVersion.config,
        });
      }
    }
  };

  const handleModelDuplication = async () => {
    const [{ name }] = await Promise.all([
      getNameSuggestion().unwrap(),
      await fillFormByModel(
        parseInt(modelId as string),
        parseInt(
          duplicationModelReference as string
        ) as modelsApi.ModelVersion['id']
      ),
    ]);

    setValue('config.name', name);
  };

  const handleModelFix = async () => {
    await fillFormByModel(
      parseInt(modelId as string),
      parseInt(
        routeParams.modelVersionId as string
      ) as modelsApi.ModelVersion['id']
    );

    //? Move to the last step
    onStepChange(steps.length - 1, 0);
  };

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

  useEffect(() => {
    currentMode == 'fix' && handleModelFix();
  }, [modelId, routeParams.modelVersionId]);

  useEffect(() => {
    currentMode == 'creation' &&
      !duplicationModelReference &&
      handleRegisteredModel();
  }, [registeredModel]);

  useEffect(() => {
    currentMode == 'duplication' && handleModelDuplication();
  }, [currentMode]);

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
                    initialElementsCount={modelEditorNodesCount}
                    onElementsCountChange={setModelEditorNodesCount}
                  />
                </TorchModelEditorContextProvider>
              </ReactFlowProvider>
            ) : (
              <Box style={{ height: '45vh' }}>
                <SklearnModelInput />
              </Box>
            )}
          </div>
          {currentMode == 'fix' && (
            <StackTrace
              stackTrace={modelVersionReference?.checkStackTrace}
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
            {['fix', 'duplication'].includes(currentMode) &&
            !modelVersionReference ? (
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
                  <span>{currentMode == 'fix' ? 'UPDATE' : 'CREATE'}</span>
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
