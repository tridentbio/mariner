import {
  TextField,
  Autocomplete,
  AutocompleteRenderInputParams,
  MenuItem,
  UseAutocompleteProps,
} from '@mui/material';
import { ColumnConfig } from 'app/rtk/generated/models';
import DataTypeChip from 'components/atoms/DataTypeChip';
import { FocusEventHandler } from 'react';

export type ValueT = { name: string; dataType: ColumnConfig['dataType'] };
type ColumnDescriptionSelectorProps<M extends boolean> = {
  'data-testid': string | undefined;
} & (M extends true
  ? {
      options: ValueT[];
      label?: string;
      error?: boolean;
      multiple?: M;
      onChange: (newVal: ValueT[]) => any;
      onBlur?: FocusEventHandler<HTMLDivElement>;
      value?: ValueT[];
      id?: string;
    }
  : {
      options: ValueT[];
      label?: string;
      error?: boolean;
      multiple?: M;
      onChange: (newVal: ValueT) => any;
      onBlur?: FocusEventHandler<HTMLDivElement>;
      value?: ValueT;
      id?: string;
    });

interface ColumnOptionProps extends AutocompleteRenderInputParams {
  label?: string;
  error?: boolean;
}

const ColumnOption = (props: ColumnOptionProps) => {
  return <TextField {...props} />;
};

const ColumnDescriptionSelector = <M extends boolean>(
  props: ColumnDescriptionSelectorProps<M>
) => {
  const handleChange: UseAutocompleteProps<
    ValueT,
    boolean,
    undefined,
    undefined
  >['onChange'] = (_event, thing) => {
    // @ts-ignore
    props.onChange(thing);
  };
  return (
    <div data-testid={props['data-testid']}>
      <Autocomplete
        id={props.id}
        sx={{ mt: 1 }}
        multiple={props.multiple}
        title={props.label}
        onChange={handleChange}
        value={props.value || (props.multiple ? [] : null)}
        options={props.options.sort((a, b) => a.name.localeCompare(b.name))}
        onBlur={props.onBlur}
        renderInput={(params) => (
          <ColumnOption label={props.label} error={props.error} {...params} />
        )}
        renderOption={(liProps, option) => (
          <MenuItem {...liProps} key={option.name}>
            {option.name}
            <DataTypeChip sx={{ ml: 'auto' }} {...option.dataType} />
          </MenuItem>
        )}
        getOptionLabel={(option) => option.name}
        isOptionEqualToValue={(optionA, optionB) => {
          return optionA?.name === optionB?.name;
        }}
      />
    </div>
  );
};

export default ColumnDescriptionSelector;
