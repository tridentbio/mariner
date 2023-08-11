import ComboBox from '@components/atoms/Select';
import { ExpandMore } from '@mui/icons-material';
import {
  Accordion,
  AccordionActions,
  AccordionDetails,
  AccordionSummary,
  Box,
  IconButton,
  SxProps,
  Theme,
} from '@mui/material';
import React, { FocusEventHandler, ReactNode, useMemo } from 'react';
import ConstructorArgInput, {
  ConstructorArgInputProps,
} from './ConstructorArgInput';
import {
  GenericPreprocessingStep,
  PreprocessingStep,
  StepValue,
} from './types';

export type PreprocessingStepSelectGetErrorFn = (
  field: 'type' | 'constructorArgs',
  value: any,
  params?: { key: string; config: ConstructorArgInputProps['arg'] }
) => boolean;

export interface PreprocessingStepSelectProps {
  value?: GenericPreprocessingStep;
  onChanges?: (step: GenericPreprocessingStep | null) => void;
  onBlur?: FocusEventHandler<HTMLDivElement>;
  getError?: PreprocessingStepSelectGetErrorFn;
  filterOptions?: (step: PreprocessingStep) => boolean;
  helperText?: string;
  options: StepValue[];
  extra?: ReactNode;
  label?: string;
  sx?: SxProps<Theme>;
}
// todo: Rename to ComponentSelect or ComponentConfig
const PreprocessingStepSelect = (props: PreprocessingStepSelectProps) => {
  const [expanded, setExpanded] = React.useState(false);

  const stepSelected = props.value;

  const getStepOption = (type: GenericPreprocessingStep['type']) => {
    return props.options.find((option) => option.type === type);
  };

  const formatStepOption = (step: StepValue) => {
    let value = step;

    if (step.constructorArgs) {
      value.constructorArgs = Object.keys(step.constructorArgs).reduce<{
        [key in keyof (typeof step)['constructorArgs']]: GenericPreprocessingStep;
      }>((acc: { [key: string]: any }, key) => {
        acc[key] =
          step.constructorArgs[
            key as keyof StepValue['constructorArgs']
            //@ts-ignore
          ]?.default;

        return acc;
      }, {});
    }

    return value as GenericPreprocessingStep;
  };

  const selectedStep = useMemo(() => {
    return stepSelected ? getStepOption(stepSelected?.type) : undefined;
  }, [stepSelected?.type]);

  const showActions =
    (stepSelected?.constructorArgs &&
      Object.keys(stepSelected.constructorArgs).length > 0) ||
    props.extra;

  return (
    <Accordion expanded={expanded} sx={props.sx}>
      <AccordionSummary>
        <ComboBox
          value={selectedStep || null}
          error={props.getError && props.getError('type', selectedStep)}
          helperText={props.helperText}
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
          isOptionEqualToValue={(option, value) => option?.type == value?.type}
          label={props.label || 'Preprocessing Step'}
          onBlur={props.onBlur}
          onChange={(_event, newValue) =>
            props.onChanges &&
            props.onChanges(newValue ? formatStepOption(newValue) : null)
          }
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
          Object.entries(stepSelected.constructorArgs).map(
            ([arg, argValue]) => {
              type ArgKey = keyof StepValue['constructorArgs'];
              const selectedArg = getStepOption(stepSelected.type)
                ?.constructorArgs[arg as ArgKey] as
                | ConstructorArgInputProps['arg']
                | undefined;

              if (!selectedArg) return;

              return (
                <Box
                  sx={{
                    margin: '16px 0',
                    flex: 1,
                    display: 'flex',
                    width: 'fit-content',
                  }}
                  key={arg}
                >
                  <ConstructorArgInput
                    value={argValue || null}
                    label={arg}
                    error={
                      props.getError &&
                      props.getError('constructorArgs', argValue, {
                        key: arg,
                        config: selectedArg,
                      })
                    }
                    helperText={props.helperText}
                    arg={selectedArg}
                    onChange={(value) => {
                      const updatedStep = stepSelected;

                      if (updatedStep.constructorArgs)
                        (updatedStep.constructorArgs as { [key: string]: any })[
                          arg
                        ] = value;

                      props.onChanges && props.onChanges(updatedStep);
                    }}
                  />
                </Box>
              );
            }
          )}
      </AccordionDetails>
    </Accordion>
  );
};

export default PreprocessingStepSelect;
