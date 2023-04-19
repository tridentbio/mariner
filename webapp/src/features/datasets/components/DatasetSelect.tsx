import { Autocomplete, AutocompleteProps, TextField } from '@mui/material';
import { Box } from '@mui/system';
import { useAppSelector } from 'app/hooks';
import { Dataset, useLazyGetMyDatasetsQuery } from 'app/rtk/generated/datasets';
import { FocusEventHandler, useEffect, useState } from 'react';

export interface DatasetSelectProps
  extends Omit<
    AutocompleteProps<Dataset, false, false, false>,
    'value' | 'onChange' | 'onBlur' | 'options' | 'renderInput'
  > {
  value?: Dataset;
  onChange?: (dataset: Dataset | null) => any;
  onBlur?: FocusEventHandler<HTMLDivElement>;
  error?: boolean;
  label?: string;
}

const DatasetSelect = (props: DatasetSelectProps) => {
  const { value: propsValue } = props;
  const [value, setValue] = useState<null | Dataset>(propsValue || null);
  const data = useAppSelector((store) => store.datasets.datasets);
  const [fetchDatasets] = useLazyGetMyDatasetsQuery();
  const datasets = data;

  useEffect(() => {
    if (propsValue) {
      setValue(propsValue);
    }
  }, [propsValue]);

  useEffect(() => {
    if (!data.length)
      fetchDatasets({
        page: 0,
        perPage: 50,
        searchByName: '',
      });
  }, [data]);
  const handleSearchChange: AutocompleteProps<
    any,
    any,
    any,
    any
  >['onInputChange'] = (_ev, input) => {
    fetchDatasets({
      page: 0,
      perPage: 50,
      searchByName: input,
    });
  };

  return (
    <Autocomplete
      {...props}
      sx={{ mt: 1 }}
      onBlur={props.onBlur}
      value={value}
      isOptionEqualToValue={(option, value) => {
        return option.id === value.id;
      }}
      // @ts-ignore
      options={datasets}
      getOptionLabel={(option) => option?.name}
      noOptionsText={'No datasets'}
      onInputChange={handleSearchChange}
      onChange={(_ev, option, reason) => {
        if (reason === 'selectOption' && option && typeof option !== 'string') {
          setValue(option);
          props.onChange && props.onChange(option);
        }
      }}
      renderOption={(params, option) => (
        <li {...params}>
          <Box>{option.name}</Box>
        </li>
      )}
      renderInput={(params) => (
        <TextField
          label={props.label || 'Dataset'}
          error={!!props.error}
          {...params}
        />
      )}
      id="dataset-select"
    />
  );
};

export default DatasetSelect;
