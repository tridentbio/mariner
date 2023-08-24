import TextField from '@mui/material/TextField';
import Autocomplete, { AutocompleteProps } from '@mui/material/Autocomplete';

type ConfiguredAutocompleteProps<T> = Omit<
  AutocompleteProps<T, false, false, false>,
  'renderInput' | 'options'
>;
export interface ComboBoxProps<T> extends ConfiguredAutocompleteProps<T> {
  options: T[];
  label: string;
  error?: boolean;
  helperText?: string;
}

export default function ComboBox<T>(props: ComboBoxProps<T>) {
  return (
    <Autocomplete
      {...props}
      disablePortal
      id="combo-box"
      options={props.options}
      onChange={(...args) => {
        if (props.onChange) props.onChange(...args);
      }}
      value={props.value}
      freeSolo={false}
      multiple={false}
      fullWidth
      renderInput={(params) => (
        <TextField
          {...params}
          label={props.label}
          error={props.error}
          helperText={props.helperText}
        />
      )}
    />
  );
}
