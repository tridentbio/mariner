import {
  Autocomplete,
  AutocompleteProps,
  Chip,
  MenuItem,
  SxProps,
  TextField,
  Theme,
} from '@mui/material';
import { Model, ModelVersion } from 'app/types/domain/models';
import { useState } from 'react';

export interface ModelVersionSelectProps {
  model: Model;
  value?: ModelVersion;
  onChange: (value: ModelVersion | undefined) => void;
  error?: boolean;
  helperText?: string;
  sx?: SxProps<Theme>;
  disableClearable?: boolean;
}

const ModelVersionSelect = ({
  model,
  value,
  onChange,
  error,
  helperText,
  sx,
  disableClearable,
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
      disableClearable={disableClearable}
      sx={sx}
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
        <MenuItem {...params} key={option.name}>
          {option.name}
          <Chip sx={{ ml: 'auto' }} label={option.config.framework} />
        </MenuItem>
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
