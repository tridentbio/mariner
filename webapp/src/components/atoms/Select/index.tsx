import TextField from '@mui/material/TextField';
import Autocomplete, { AutocompleteProps } from '@mui/material/Autocomplete';

type ConfiguredAutocompleteProps<T> = Omit<
  AutocompleteProps<T, false, false, false>,
  'renderInput' | 'options'
>;
export interface ComboBoxProps<T> extends ConfiguredAutocompleteProps<T> {
  options: T[];
  label: string;
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
      sx={{ width: 300 }}
      renderInput={(params) => <TextField {...params} label={props.label} />}
    />
  );
}
