import { Autocomplete, TextField } from '@mui/material';
import { Box } from '@mui/system';
import { useEffect, useMemo, useState } from 'react';
import { Model } from 'app/types/domain/models';
import { modelsApi } from 'app/rtk/models';
import { useSearchParams } from 'react-router-dom';

export interface ModelsSelectProps {
  value?: Model;
  onChange: (value: Model | null) => void;
  error?: boolean;
  helperText?: string;
}

const ModelsSelect = ({
  value,
  onChange,
  error,
  helperText,
}: ModelsSelectProps) => {
  const [searchParams] = useSearchParams();
  const modelId = searchParams.get('modelId');
  const [searchName, setSearchName] = useState<string>('');
  const { data } = modelsApi.useGetModelsOldQuery({
    q: searchName,
    perPage: 1000,
  });
  const dependencyData = JSON.stringify(data);
  const models = useMemo(() => data?.data || [], [dependencyData]);

  useEffect(() => {
    if (!!models && modelId) {
      const foundModel = models.find(({ id }) => id === Number(modelId));
      if (foundModel) onChange(foundModel);
    }
  }, [dependencyData]);

  return (
    <Autocomplete
      value={value || null}
      isOptionEqualToValue={(option, value) => {
        return option.id === value.id;
      }}
      getOptionLabel={(option) => option.name}
      options={models}
      noOptionsText={'No model found'}
      onInputChange={(_e, value) => setSearchName(value)}
      onChange={(_ev, option) => {
        onChange(option);
      }}
      renderOption={(params, option) => (
        <li {...params}>
          <Box>{option.name}</Box>
        </li>
      )}
      renderInput={(params) => (
        <TextField
          helperText={helperText}
          error={!!error}
          label="Model"
          {...params}
        />
      )}
      id="model-select"
    />
  );
};

export default ModelsSelect;
