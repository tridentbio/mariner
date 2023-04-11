import { Autocomplete, AutocompleteProps, TextField } from '@mui/material';
import { Box } from '@mui/system';
import { useState } from 'react';
import { Model, ModelVersion } from 'app/types/domain/models';

export interface ModelVersionSelectProps {
  model: Model;
  value?: ModelVersion;
  onChange: (value: ModelVersion | undefined) => void;
  error?: boolean;
  helperText?: string;
}

const ModelVersionSelect = ({
  model,
  value,
  onChange,
  error,
  helperText,
}: ModelVersionSelectProps) => {
  const [options, setOptions] = useState(model.versions);

  const handleSearchChange: AutocompleteProps<
    any,
    any,
    any,
    any
  >['onInputChange'] = (_ev, input) => {
    setOptions(
      model.versions.filter(
        (option) =>
          (option.name && option.name.includes(input)) ||
          (option.description && option.description.includes(input))
      )
    );
  };

  return (
    <Autocomplete
      value={value || null}
      isOptionEqualToValue={(option, value) => {
        return option.id === value.id;
      }}
      getOptionLabel={(option) => option.name}
      options={model.versions}
      noOptionsText={'No datasets'}
      onInputChange={handleSearchChange}
      onChange={(_ev, option) => {
        onChange(option || undefined);
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
          label="Model Version"
          {...params}
        />
      )}
      id="model-version-select"
    />
  );
};

export default ModelVersionSelect;
