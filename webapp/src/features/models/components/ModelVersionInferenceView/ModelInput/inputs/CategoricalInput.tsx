import {
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  FormLabel,
} from '@mui/material';

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
      <FormLabel id="demo-radio-buttons-group-label">{label}:</FormLabel>
      <Select
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
