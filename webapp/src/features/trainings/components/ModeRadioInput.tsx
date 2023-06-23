import {
  Box,
  FormControlLabel,
  InputLabel,
  Radio,
  RadioGroup,
  Tooltip,
} from '@mui/material';
import { NewTraining } from 'app/types/domain/experiments';
import React from 'react';
import { Controller, ControllerProps, useFormContext } from 'react-hook-form';

type ModeRadioInputProps = {
  fieldName: ControllerProps<NewTraining, any>['name'];
};

const ModeRadioInput: React.FC<ModeRadioInputProps> = ({ fieldName }) => {
  const { control } = useFormContext<NewTraining>();
  return (
    <Box sx={{ mb: 1 }}>
      <InputLabel htmlFor={fieldName}>Mode </InputLabel>
      <Controller
        control={control}
        name={fieldName}
        render={({ field }) => (
          <Tooltip
            title={'Minimize or maximize monitored value'}
            placement={'right'}
          >
            <RadioGroup
              row
              aria-labelledby={`${fieldName}-radio-buttons-group`}
              {...field}
              sx={{ display: 'inline-block' }}
            >
              <FormControlLabel
                label={'Minimize'}
                value={'min'}
                control={<Radio />}
              />
              <FormControlLabel
                label={'Maximize'}
                value={'max'}
                control={<Radio />}
              />
            </RadioGroup>
          </Tooltip>
        )}
      />
    </Box>
  );
};

export default ModeRadioInput;
