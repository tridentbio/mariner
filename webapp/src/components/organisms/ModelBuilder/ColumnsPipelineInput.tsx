import { ModelCreate } from '@app/rtk/generated/models';
import { DataTypeGuard } from '@app/types/domain/datasets';
import { Text } from '@components/molecules/Text';
import ColumnConfigurationAccordion from '@features/models/components/ColumnConfigurationView/ColumnConfigAccordion';
import DeleteOutline from '@mui/icons-material/DeleteOutline';
import { Button, Chip, Divider, IconButton } from '@mui/material';
import { useEffect } from 'react';
import {
  Controller,
  ControllerRenderProps,
  useFieldArray,
  useFormContext,
} from 'react-hook-form';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import {
  GenericPreprocessingStep,
  SimpleColumnConfig,
  StepValue,
} from './types';
import {
  StepFormFieldError,
  getColumnConfigTestId,
  getStepSelectError,
} from './utils';
import useModelBuilder from './hooks/useModelBuilder';

export interface ColumnsPipelineInputProps {
  column: {
    config: SimpleColumnConfig;
    index: number;
    type: 'featureColumns' | 'targetColumns';
  };
  featurizerOptions: StepValue[];
  transformOptions: StepValue[];
}

export default function ColumnsPipelineInput(props: ColumnsPipelineInputProps) {
  const { featurizerOptions, transformOptions, column } = props;
  const { control, trigger } = useFormContext<ModelCreate>();
  const { editable, defaultExpanded } = useModelBuilder();

  //? Being used as a single item array on featurizers <PreprocessingStepSelect /> list
  const featurizersOptionsField = useFieldArray({
    control,
    name: `config.dataset.${column.type}.${column.index}.featurizers`,
  });

  const transformsOptionsField = useFieldArray({
    control,
    name: `config.dataset.${column.type}.${column.index}.transforms`,
  });

  const emptyStep = {
    type: '',
    constructorArgs: {},
  } as GenericPreprocessingStep;

  const addFeaturizer = () => {
    featurizersOptionsField.append(emptyStep);
  };

  const addTransform = () => {
    transformsOptionsField.append(emptyStep);
  };

  const deleteColumnTransform = (stepIndex: number) => {
    transformsOptionsField.remove(stepIndex);
  };

  const onStepSelect = <
    FieldNames extends `config.dataset.${(typeof column)['type']}.${number}.${
      | 'transforms'
      | 'featurizers'}.${number}`
  >(
    field: ControllerRenderProps<ModelCreate, FieldNames>,
    newValue: GenericPreprocessingStep | null
  ) => {
    field.onChange(newValue ?? emptyStep);

    trigger(field.name);
  };

  useEffect(() => {
    if (
      !featurizersOptionsField.fields.length &&
      !DataTypeGuard.isNumericalOrQuantity(column.config.dataType)
    )
      addFeaturizer();
  }, []);

  return (
    <>
      <ColumnConfigurationAccordion
        key={column.config.name}
        testId={getColumnConfigTestId(column.config)}
        dataType={column.config.dataType}
        name={column.config.name}
        defaultExpanded={defaultExpanded}
      >
        {!DataTypeGuard.isNumericalOrQuantity(column.config.dataType) &&
          featurizersOptionsField.fields.map((step, stepIndex) => (
            <>
              <Divider textAlign="center" sx={{ marginBottom: 2 }}>
                <Chip
                  label="Featurizers"
                  data-testid={`${getColumnConfigTestId(
                    column.config
                  )}-featurizer-label`}
                />
              </Divider>
              <Controller
                key={step.id}
                control={control}
                name={`config.dataset.${column.type}.${column.index}.featurizers.${stepIndex}`}
                render={({ field, fieldState: { error } }) => (
                  <PreprocessingStepSelect
                    testId={`${getColumnConfigTestId(
                      column.config
                    )}-featurizer-${stepIndex}`}
                    sx={{
                      mb: 3,
                      borderRadius: 2,
                      border: 'none',
                      boxShadow: '0px 0px 1px',
                    }}
                    options={featurizerOptions}
                    getError={getStepSelectError(
                      () => error as StepFormFieldError | undefined
                    )}
                    value={field.value || null}
                    helperText={error?.message}
                    onBlur={field.onBlur}
                    onChanges={(step) => onStepSelect(field, step)}
                  />
                )}
              />
            </>
          ))}

        <Divider textAlign="center" sx={{ marginBottom: 2 }}>
          <Chip label="Transforms" />
        </Divider>
        {transformsOptionsField.fields.map((step, stepIndex) => (
          <Controller
            key={step.id}
            control={control}
            name={`config.dataset.${column.type}.${column.index}.transforms.${stepIndex}`}
            render={({ field, fieldState: { error } }) => {
              return (
                <PreprocessingStepSelect
                  testId={`${getColumnConfigTestId(
                    column.config
                  )}-transform-${stepIndex}`}
                  options={transformOptions}
                  getError={getStepSelectError(
                    () => error as StepFormFieldError | undefined
                  )}
                  helperText={error?.message}
                  onBlur={field.onBlur}
                  onChanges={(updatedStep) => onStepSelect(field, updatedStep)}
                  value={field.value || null}
                  extra={
                    editable ? (
                      <IconButton
                        onClick={() => deleteColumnTransform(stepIndex)}
                      >
                        <DeleteOutline />
                      </IconButton>
                    ) : null
                  }
                />
              );
            }}
          />
        ))}
        {editable && (
          <Button
            variant="contained"
            sx={{ mt: 1 }}
            onClick={() => addTransform()}
          >
            ADD
          </Button>
        )}
      </ColumnConfigurationAccordion>
    </>
  );
}
