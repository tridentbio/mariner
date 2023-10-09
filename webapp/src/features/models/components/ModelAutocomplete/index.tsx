import { useEffect, useState, SyntheticEvent } from 'react';
import {
  Autocomplete,
  TextField,
  AutocompleteProps,
  MenuItem,
  Chip,
  CircularProgress,
  AutocompleteInputChangeReason,
} from '@mui/material';
import { modelsApi } from 'app/rtk/models';
import { debounce } from 'utils';

interface ModelAutoCompleteProps
  extends Partial<
    Omit<AutocompleteProps<any, any, any, any>, 'onChange' | 'value'>
  > {
  label?: string;
  value?: { name: string };
  error?: boolean;
  onChange?: (newVal?: {
    name: string;
    new?: boolean;
    id?: number;
    loading?: boolean;
  }) => any;
}

const ModelAutoComplete = ({
  onChange,
  value,
  label,
  error,
  ...selectProps
}: ModelAutoCompleteProps) => {
  const [getModels, { data }] = modelsApi.useLazyGetModelsOldQuery({});
  const [inputVal, setInputVal] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    getModels({ q: '', page: 0, perPage: 15 });
  }, []);

  const handleInputChange = debounce(
    (
      _event: SyntheticEvent<Element, Event>,
      value: string,
      _reason: AutocompleteInputChangeReason
    ) => {
      setInputVal(value);
      if (!isLoading) setIsLoading(true);
      getModels({ q: value, perPage: 10, page: 0 }).finally(() =>
        setIsLoading(false)
      );
    }
  );

  let options = (data?.data || []) as ({ name: string } & {
    new?: boolean;
    loading?: boolean;
  })[];

  if (
    inputVal &&
    (!data?.data?.length ||
      !data.data.map((model) => model.name).includes(inputVal))
  ) {
    options = options.concat([
      { name: inputVal, new: true, loading: isLoading },
    ]);
  }

  return (
    <Autocomplete
      loading={isLoading}
      onInputChange={handleInputChange}
      options={options}
      freeSolo
      onChange={(_event, t, a) => {
        if (!onChange) return;
        if (a === 'selectOption') onChange(t || undefined);
        else if (a === 'clear') onChange();
        else if (a === 'createOption') {
          onChange({ name: t });
        }
      }}
      value={value?.name ? value : null}
      getOptionLabel={(option) => {
        return option?.name || '';
      }}
      renderOption={(props, option) => (
        <MenuItem {...props}>
          {option.name}
          {option.loading && (
            <CircularProgress sx={{ ml: 3, width: 10, height: 10 }} />
          )}
          {!option?.loading && option.new && (
            <Chip color="success" sx={{ ml: 3 }} label="new" />
          )}
          {!option?.loading && !option?.new && (
            <Chip sx={{ ml: 3 }} label="existing" />
          )}
        </MenuItem>
      )}
      isOptionEqualToValue={(a, b) => a?.name === b?.name}
      renderInput={(params) => (
        <TextField error={error} {...params} label={label} />
      )}
      {...selectProps}
    />
  );
};

export default ModelAutoComplete;
