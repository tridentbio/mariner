import ComboBox from '@components/atoms/Select';
import { ExpandMore } from '@mui/icons-material';
import {
  Accordion,
  AccordionActions,
  AccordionDetails,
  AccordionProps,
  AccordionSummary,
  Box,
  IconButton,
} from '@mui/material';
import React, { FocusEventHandler, ReactNode, useEffect, useMemo } from 'react';
import ConstructorArgInput, {
  ConstructorArgInputProps,
} from './ConstructorArgInput';
import useModelBuilder from './hooks/useModelBuilder';
import {
  GenericPreprocessingStep,
  PreprocessingStep,
  StepValue,
} from './types';
import { getStepValueLabelData } from './utils';

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
  sx?: AccordionProps['sx'];
  testId?: string;
}
// todo: Rename to ComponentSelect or ComponentConfig
const PreprocessingStepSelect = (props: PreprocessingStepSelectProps) => {
  const [expanded, setExpanded] = React.useState(false);
  const { editable, defaultExpanded } = useModelBuilder();

  const stepSelected = props.value;

  useEffect(() => {
    setExpanded(defaultExpanded);
  }, []);

  const getStepOption = (type: GenericPreprocessingStep['type']) => {
    return props.options.find((option) => option.type === type);
  };

  const formatStepOption = (step: StepValue) => {
    let value = Object.assign({}, step);

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

  const selectedStepOption = useMemo(() => {
    return stepSelected ? getStepOption(stepSelected?.type) : undefined;
  }, [stepSelected, props.options]);

  const showActions =
    (stepSelected?.constructorArgs &&
      Object.keys(stepSelected.constructorArgs).length > 0) ||
    props.extra;

  const hasConstructorArgs = !!Object.keys(stepSelected?.constructorArgs || {})
    .length;

  useEffect(() => {
    if (!hasConstructorArgs) setExpanded(false);
  }, [stepSelected?.constructorArgs]);

  return (
    <Accordion
      disableGutters
      expanded={expanded}
      sx={props.sx}
      data-testid={props.testId}
    >
      <AccordionSummary>
        <ComboBox
          className="step-select"
          value={selectedStepOption || null}
          error={props.getError && props.getError('type', stepSelected)}
          helperText={props.helperText}
          options={props.options}
          getOptionLabel={(option) => {
            const labelData = getStepValueLabelData(option.type);

            return labelData
              ? `${labelData.class} (${labelData.lib})`
              : 'Select one';
          }}
          isOptionEqualToValue={(option, value) => option?.type == value?.type}
          label={props.label || 'Preprocessing Step'}
          onBlur={props.onBlur}
          onChange={(_event, newValue) =>
            props.onChanges &&
            props.onChanges(newValue ? formatStepOption(newValue) : null)
          }
          disabled={!editable}
          sx={{ pointerEvents: editable ? 'auto' : 'none' }}
        />
        {showActions && (
          <AccordionActions>
            <>
              {hasConstructorArgs && (
                <IconButton
                  data-testid={`${props.testId}-action-btn`}
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
        {hasConstructorArgs &&
          Object.entries(
            stepSelected?.constructorArgs as { [key: string]: any }
          ).map(([arg, argValue]) => {
            type ArgKey = keyof StepValue['constructorArgs'];
            const selectedArg = selectedStepOption?.constructorArgs[
              arg as ArgKey
            ] as ConstructorArgInputProps['arg'] | undefined;

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
                    const updatedStep =
                      stepSelected as GenericPreprocessingStep;

                    if (updatedStep.constructorArgs)
                      (updatedStep.constructorArgs as { [key: string]: any })[
                        arg
                      ] = value;

                    props.onChanges && props.onChanges(updatedStep);
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
