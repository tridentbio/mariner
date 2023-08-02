import React, { ReactNode } from 'react';
import ComboBox from '@components/atoms/Select';
import {
  PreprocessingStep,
  PreprocessingStepSelectProps,
  StepValue,
} from './types';
import ConstructorArgInput from './ConstructorArgInput';
import {
  Accordion,
  AccordionActions,
  AccordionDetails,
  AccordionSummary,
  Box,
  IconButton,
} from '@mui/material';
import ArrowDownward from '@mui/icons-material/ArrowDownward';
import { ExpandMore } from '@mui/icons-material';

export interface PreprocessingStepSelectProps {
  value?: PreprocessingStep;
  onChange: (step?: PreprocessingStep) => any;
  filterOptions?: (step: PreprocessingStep) => boolean;
  error?: boolean;
  helperText?: string;
  options: StepValue[];
  extra?: ReactNode;
  label?: string;
}
// todo: Rename to ComponentSelect or ComponentConfig
const PreprocessingStepSelect = (props: PreprocessingStepSelectProps) => {
  const [stepSelected, setStepSelected] = React.useState<StepValue | undefined>(
    props.options.find((opt) => opt.type && opt.type === props.value?.type)
  );

  const [constructorArgs, setConstructorArgs] = React.useState<
    Record<string, any>
  >({});

  const [expanded, setExpanded] = React.useState(false);
  const showActions =
    (stepSelected?.constructorArgs &&
      Object.keys(stepSelected.constructorArgs).length > 0) ||
    props.extra;
  return (
    <Accordion expanded={expanded}>
      <AccordionSummary>
        <ComboBox
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
          label={props.label || "Preprocessing Step"}
          value={stepSelected || null}
          onChange={(_event, newValue) => {
            if (newValue) setStepSelected(newValue);
            else setStepSelected(undefined);
            const defaultConstructorArgs = {};
            for (const [key, value] of Object.entries(
              newValue?.constructorArgs || {}
            )) {
              // @ts-ignore
              defaultConstructorArgs[key] = value.default;
            }
            setConstructorArgs(defaultConstructorArgs);
            props.onChange &&
              props.onChange({
                type: newValue?.type,
                constructorArgs: defaultConstructorArgs,
              } as any);
          }}
          onClick={(event) => event.stopPropagation()}
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
          Object.entries(stepSelected.constructorArgs).map(([key, value]) => {
            return (
              <Box sx={{ margin: '16px 0', flex: 1, display: 'flex', width: 'fit-content' }} key={key}>
                <ConstructorArgInput
                  label={key}
                  arg={value}
                  value={constructorArgs[key]}
                  onChange={(val) => {
                    const newVal = {
                      ...constructorArgs,
                      [key]: val,
                    };
                    setConstructorArgs(newVal);
                    // @ts-ignore
                    props.onChange &&
                      props.onChange({
                        type: stepSelected.type,
                        constructorArgs: newVal,
                      });
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
