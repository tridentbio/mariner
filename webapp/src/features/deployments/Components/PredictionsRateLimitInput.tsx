import {
  Box,
  InputAdornment,
  InputLabel,
  MenuItem,
  TextField,
} from '@mui/material';
import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import { required } from '@utils/reactFormRules';
import { ERateLimitUnits } from '../types';
import CustomRateLimitSlider from './CustomRateLimitSlider';

type PredictionsRateLimitInputProps = {
  fieldStyle: Record<string, any>;
};

const units = [
  {
    value: ERateLimitUnits.MINUTE,
    label: 'minute',
  },
  {
    value: ERateLimitUnits.HOUR,
    label: 'hour',
  },
  {
    value: ERateLimitUnits.DAY,
    label: 'day',
  },
  {
    value: ERateLimitUnits.MONTH,
    label: 'month',
  },
];

const PredictionsRateLimitInput: React.FC<PredictionsRateLimitInputProps> = ({
  fieldStyle,
}) => {
  const { control } = useFormContext();
  return (
    <Box>
      <InputLabel>Predictions Rate Limit (per user)</InputLabel>

      <Box sx={{ ...fieldStyle, display: 'flex', gap: 2 }}>
        <CustomRateLimitSlider />
        <Controller
          name="predictionRateLimitUnit"
          rules={{ ...required }}
          control={control}
          render={({ field: { onChange, value }, fieldState: { error } }) => {
            return (
              <TextField
                sx={{ width: '250px' }}
                label="Per Unit"
                select
                error={!!error}
                helperText={error?.message}
                value={value}
                onChange={(e) => {
                  onChange({ target: { value: e.target.value } });
                }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">/</InputAdornment>
                  ),
                }}
              >
                {units.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </TextField>
            );
          }}
        />
      </Box>
    </Box>
  );
};

export default PredictionsRateLimitInput;
