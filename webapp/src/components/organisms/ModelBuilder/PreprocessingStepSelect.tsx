import React from 'react';
import ComboBox from '@components/atoms/Select';
import {
  PreprocessingStepSelectProps,
  StepValue,
  TypeIdentifier,
} from './types';
import ConstructorArgInput from './ConstructorArgInput';
import {
  Accordion,
  AccordionActions,
  AccordionDetails,
  AccordionSummary,
  IconButton,
} from '@mui/material';
import ArrowDownward from '@mui/icons-material/ArrowDownward';

const getType = (python_type: any): TypeIdentifier | undefined => {
  if (typeof python_type !== 'string') return;
  if (python_type.includes('int') || python_type.includes('float'))
    return 'number';
  else if (python_type.includes('bool')) return 'bool';
};

const PreprocessingStepSelect = (props: PreprocessingStepSelectProps) => {
  const [stepSelected, setStepSelected] = React.useState<StepValue | undefined>(
    props.options.find((opt) => opt.type && opt.type === props.value?.type)
  );

  const [constructorArgs, setConstructorArgs] = React.useState<
    Record<string, any>
  >({});

  const [expanded, setExpanded] = React.useState(false);
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
          label="Preprocessing Step"
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
        <AccordionActions>
          {stepSelected && stepSelected.constructorArgs && (
            <IconButton onClick={() => setExpanded((expanded) => !expanded)}>
              <ArrowDownward
                sx={{
                  transform: expanded ? 'rotate(180deg)' : undefined,
                }}
              />
            </IconButton>
          )}
        </AccordionActions>
      </AccordionSummary>
      <AccordionDetails>
        {stepSelected &&
          stepSelected.constructorArgs &&
          Object.entries(stepSelected.constructorArgs).map(([key, value]) => {
            return (
              <div key={key}>
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
              </div>
            );
          })}
      </AccordionDetails>
    </Accordion>
  );
};

export default PreprocessingStepSelect;
