import ComboBox from '@components/atoms/Select';
import { ExpandMore } from '@mui/icons-material';
import {
  Accordion,
  AccordionActions,
  AccordionDetails,
  AccordionSummary,
  IconButton,
} from '@mui/material';
import React, { ReactNode } from 'react';
import {
  Controller,
  ControllerRenderProps,
  useFormContext,
} from 'react-hook-form';
import ConstructorArgInput, {
  ConstructorArgInputProps,
} from './ConstructorArgInput';
import {
  DatasetConfigPreprocessing,
  PreprocessingStep,
  StepValue,
} from './types';

export interface PreprocessingStepSelectProps {
  value?: PreprocessingStep;
  filterOptions?: (step: PreprocessingStep) => boolean;
  error?: boolean;
  helperText?: string;
  options: StepValue[];
  extra?: ReactNode;
  stepFieldName: `${'featureColumns' | 'targetColumns'}.${number}.${
    | 'transforms'
    | 'featurizers'}.${number}`;
}
const PreprocessingStepSelect = (props: PreprocessingStepSelectProps) => {
  const [expanded, setExpanded] = React.useState(false);

  const { control, watch, setValue, trigger } =
    useFormContext<DatasetConfigPreprocessing>();

  const stepSelected = watch(`${props.stepFieldName}`);

  const showActions =
    (stepSelected?.constructorArgs &&
      Object.keys(stepSelected.constructorArgs).length > 0) ||
    props.extra;

  const onTypeSelect = (
    field: ControllerRenderProps<DatasetConfigPreprocessing, any>,
    newValue: StepValue | null
  ) => {
    field.onChange(newValue?.type);

    setValue(
      `${props.stepFieldName}.constructorArgs`,
      newValue?.constructorArgs as object
    );

    refreshConstructorArgsValidation();
  };

  const onConstructorArgChange = (
    field: ControllerRenderProps<DatasetConfigPreprocessing, any>,
    newValue: string | number | boolean
  ) => {
    field.onChange({
      ...field.value,
      default: newValue,
    });

    refreshConstructorArgsValidation();
  };

  const refreshConstructorArgsValidation = () =>
    trigger(`${props.stepFieldName}.constructorArgs`);

  return (
    <Accordion expanded={expanded}>
      <AccordionSummary>
        <Controller
          control={control}
          name={`${props.stepFieldName}.type`}
          render={({ field, fieldState: { error } }) => {
            const value = props.options.find(
              (option) => option.type === field.value
            );

            return (
              <ComboBox
                {...field}
                value={value || null}
                error={!!error}
                helperText={error?.message}
                options={props.options}
                getOptionLabel={(option) => {
                  if (option.type) {
                    const parts = option.type.split('.');
                    const lib = parts[0];
                    const class_ = parts.at(-1);
                    return `${class_} (${lib})`;
                  }
                  return 'Select one';
                }}
                label="Preprocessing Step"
                onChange={(_event, newValue) =>
                  onTypeSelect(field, newValue as StepValue)
                }
              />
            );
          }}
        />
        {showActions && (
          <AccordionActions>
            <>
              {stepSelected?.constructorArgs && (
                <IconButton
                  onClick={() => setExpanded((expanded) => !expanded)}
                >
                  <ExpandMore
                    sx={{
                      transform: expanded ? 'rotate(180deg)' : undefined,
                      transition: 'transform 0.2s',
                    }}
                  />
                </IconButton>
              )}
              {props.extra || null}
            </>
          </AccordionActions>
        )}
      </AccordionSummary>
      <AccordionDetails>
        {stepSelected &&
          stepSelected.constructorArgs &&
          Object.entries(stepSelected.constructorArgs).map(([key, arg]) => {
            return (
              <Controller
                key={key}
                control={control}
                name={`${props.stepFieldName}.constructorArgs.${key}` as any}
                render={({ field, fieldState: { error } }) => (
                  <ConstructorArgInput
                    value={field.value.default || arg.default}
                    label={key}
                    error={!!error}
                    helpText={error?.message}
                    arg={arg as ConstructorArgInputProps['arg']}
                    onChange={(value) => onConstructorArgChange(field, value)}
                  />
                )}
              />
            );
          })}
      </AccordionDetails>
    </Accordion>
  );
};

export default PreprocessingStepSelect;
