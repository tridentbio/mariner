import { DataTypeGuard } from '@app/types/domain/datasets';
import { Text } from '@components/molecules/Text';
import ColumnConfigurationAccordion from '@features/models/components/ColumnConfigurationView/ColumnConfigAccordion';
import DeleteOutline from '@mui/icons-material/DeleteOutline';
import { Button, IconButton } from '@mui/material';
import {
  Controller,
  ControllerRenderProps,
  FieldError,
  useFieldArray,
  useFormContext,
} from 'react-hook-form';
import PreprocessingStepSelect, {
  PreprocessingStepSelectGetErrorFn,
} from './PreprocessingStepSelect';
import {
  DatasetConfigPreprocessing,
  GenericPreprocessingStep,
  SimpleColumnConfig,
  StepValue,
} from './types';
import { useEffect } from 'react';

export interface ColumnsPipelineInputProps {
  column: {
    config: SimpleColumnConfig;
    index: number;
    type: 'featureColumns' | 'targetColumns';
  };
  featurizerOptions: StepValue[];
  transformOptions: StepValue[];
}

type StepFormFieldError = {
  type: FieldError;
  constructorArgs: { [key: string]: FieldError };
};

export default function ColumnsPipelineInput(props: ColumnsPipelineInputProps) {
  const { featurizerOptions, transformOptions, column } = props;
  const { control, trigger } = useFormContext<DatasetConfigPreprocessing>();

  //? Being used as a single item array on featurizers <PreprocessingStepSelect /> list
  const featurizersOptionsField = useFieldArray({
    control,
    name: `${column.type}.${column.index}.featurizers`,
  });

  const transformsOptionsField = useFieldArray({
    control,
    name: `${column.type}.${column.index}.transforms`,
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
    FieldNames extends `${(typeof column)['type']}.${number}.${
      | 'transforms'
      | 'featurizers'}.${number}`
  >(
    field: ControllerRenderProps<DatasetConfigPreprocessing, FieldNames>,
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

  const getStepError = (
    fieldError?: StepFormFieldError
  ): PreprocessingStepSelectGetErrorFn => {
    return (type, key) => {
      if (!fieldError) return false;

      switch (type) {
        case 'type':
          return !!fieldError.type;
        default: {
          const constructorArgsError = fieldError.constructorArgs;

          return constructorArgsError && !!constructorArgsError[key as string];
        }
      }
    };
  };

  return (
    <>
      <ColumnConfigurationAccordion
        key={column.config.name}
        dataType={column.config.dataType}
        name={column.config.name}
      >
        {!DataTypeGuard.isNumericalOrQuantity(column.config.dataType) &&
          featurizersOptionsField.fields.map((step, stepIndex) => (
            <>
              <Text sx={{ width: '100%' }}>Featurizers:</Text>
              <Controller
                key={step.id}
                control={control}
                name={`${column.type}.${column.index}.featurizers.${stepIndex}`}
                render={({ field, fieldState: { error } }) => (
                  <PreprocessingStepSelect
                    options={featurizerOptions}
                    // @ts-ignore
                    getError={getStepError(error as StepFormFieldError)}
                    value={field.value || null}
                    helperText={error?.message}
                    onChanges={(step) => onStepSelect(field, step)}
                  />
                )}
              />
            </>
          ))}

        <Text sx={{ width: '100%' }}>Transforms:</Text>
        {transformsOptionsField.fields.map((step, stepIndex) => (
          <Controller
            key={step.id}
            control={control}
            name={`${column.type}.${column.index}.transforms.${stepIndex}`}
            render={({ field, fieldState: { error } }) => {
              return (
                <PreprocessingStepSelect
                  options={transformOptions}
                  // @ts-ignore
                  getError={getStepError(error as StepFormFieldError)}
                  helperText={error?.message}
                  onChanges={(updatedStep) => onStepSelect(field, updatedStep)}
                  value={field.value || null}
                  extra={
                    <IconButton
                      onClick={() => deleteColumnTransform(stepIndex)}
                    >
                      <DeleteOutline />
                    </IconButton>
                  }
                />
              );
            }}
          />
        ))}
        <Button
          variant="contained"
          sx={{ mt: 1 }}
          onClick={() => addTransform()}
        >
          ADD
        </Button>
      </ColumnConfigurationAccordion>
    </>
  );
}
