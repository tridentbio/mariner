import { MenuItem, FormControl, InputLabel, Select } from '@mui/material';

interface CategoricalInputProps<T extends {}> {
  options: T[];
  getLabel: (thing: T) => string;
  value: T;
  onChange: (v: string) => any;
  label: string;
}

const CategoricalInput = <T extends {}>({
  options,
  getLabel,
  value,
  onChange,
  label,
}: CategoricalInputProps<T>) => {
  return (
    <FormControl sx={{ width: 300 }}>
      <InputLabel id="categories-label">{label}</InputLabel>
      <Select
        label={label}
        value={getLabel(value)}
        labelId="categories-label"
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((opt) => (
          <MenuItem value={getLabel(opt)} key={getLabel(opt)}>
            {getLabel(opt)}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};

export default CategoricalInput;
