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
import React, { FocusEventHandler, ReactNode } from 'react';
import ConstructorArgInput, {
  ConstructorArgInputProps,
} from './ConstructorArgInput';
import {
  GenericPreprocessingStep,
  PreprocessingStep,
  StepValue,
} from './types';

export type PreprocessingStepSelectGetErrorFn = (
  field: keyof GenericPreprocessingStep,
  key?: string
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

  const showActions =
    (stepSelected?.constructorArgs &&
      Object.keys(stepSelected.constructorArgs).length > 0) ||
    props.extra;

  return (
    <Accordion expanded={expanded} sx={props.sx}>
      <AccordionSummary>
        <ComboBox
          value={stepSelected?.type ? stepSelected : null}
          error={props.getError && props.getError('type')}
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
            props.onChanges && props.onChanges(newValue)
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
          Object.entries(stepSelected.constructorArgs).map(([key, arg]) => {
            return (
              <Box
                sx={{
                  margin: '16px 0',
                  flex: 1,
                  display: 'flex',
                  width: 'fit-content',
                }}
                key={key}
              >
                <ConstructorArgInput
                  key={key}
                  value={arg.default || null}
                  label={key}
                  error={
                    props.getError && props.getError('constructorArgs', key)
                  }
                  helperText={props.helperText}
                  arg={arg as ConstructorArgInputProps['arg']}
                  onChange={(value) => {
                    let updatedConstructorArgs = stepSelected;

                    const argKey =
                      key as keyof GenericPreprocessingStep['constructorArgs'];

                    // @ts-ignore
                    updatedConstructorArgs.constructorArgs[argKey] = {
                      // @ts-ignore
                      ...updatedConstructorArgs.constructorArgs[argKey],
                      default: value,
                    };

                    props.onChanges && props.onChanges(updatedConstructorArgs);
                  }}
                />
              </Box>
            );
          })}
      </AccordionDetails>
    </Accordion>
  );
};

export default PreprocessingStepSelect;
