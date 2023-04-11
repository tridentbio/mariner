import React, { useMemo } from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import { required } from 'utils/reactFormRules';

import { RateLimitSlider } from './styles';

const marks = [
  {
    value: 0,
  },
  {
    value: 50,
  },
  {
    value: 100,
  },
];

const CustomRateLimitSlider: React.FC = () => {
  const { control, watch } = useFormContext();
  const defaultValue = useMemo(() => watch('predictionRateLimitValue'), []);
  return (
    <Controller
      name="predictionRateLimitValue"
      rules={{ ...required }}
      control={control}
      render={({
        field: { onChange, value: fieldValue },
        fieldState: { error },
      }) => {
        return (
          <RateLimitSlider
            aria-label="rate limit slider"
            defaultValue={defaultValue}
            onChangeCommitted={(_e, value) => onChange({ target: { value } })}
            marks={marks}
            valueLabelDisplay="on"
            sx={{ width: '100%', mt: 4 }}
          />
        );
      }}
    />
  );
};

export default CustomRateLimitSlider;
