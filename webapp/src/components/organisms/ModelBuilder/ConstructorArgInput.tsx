import { FormControl, FormControlLabel, Input, InputLabel, MenuItem, Select, Switch, TextField } from '@mui/material';
import { TypeIdentifier } from './types';

interface ArgOption {
  key: string
  label?: string
  latex?: string
}
interface ConstructorArgInputProps {
  arg: {
    type: TypeIdentifier;
    default: any;
    options?: string | ArgOption[];
    required?: boolean;
  };
  value: any;
  label: string;
  onChange: (value: any) => void;
}

const getLabel = (argOption: string | ArgOption) => {
  if (typeof argOption === 'string') {
    return argOption;
  } else {
    return argOption.label || argOption.key;
  }
}
const ConstructorArgInput = ({
  arg,
  value,
  label,
  onChange,
}: ConstructorArgInputProps) => {
  const formControlStyle = { minWidth: 200 }
  const inputId = `arg-option-${label}`
  const variant = "standard"
  if (arg.options && arg.options.length) {
    return (
      <FormControl sx={formControlStyle}>
        <InputLabel variant={variant} htmlFor="arg-option">{label}</InputLabel>
        <Select
          variant={variant}
          id={inputId}
          defaultValue={arg.default}
          onChange={(event) => onChange(event.target.value)}
        >
          {arg.options.map((option) => (
            <MenuItem key={option.key} sx={{ minWidth: '100%' }} value={option.key}>
              {getLabel(option)}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    );
  }
  else if (arg.type === 'boolean') {
    return (
      <FormControl sx={formControlStyle}>
        <FormControlLabel control={<Switch
          defaultValue={arg.default}
          checked={value}
          onChange={(event) => onChange(event.target.checked)}
        />} label={label}  labelPlacement="end"/>
      </FormControl>
    );
  } else if (arg.type === 'string') {
    return (
      <FormControl sx={formControlStyle}>
        <InputLabel variant={variant} htmlFor={inputId}>{label}</InputLabel>
        <Input
          id={inputId}
          defaultValue={arg.default}
          onChange={(event) => onChange(event.target.value)}
        />
      </FormControl>
    );
  } else if (arg.type === 'number') {
    return (
      <FormControl sx={formControlStyle}>
        <InputLabel variant={variant} htmlFor={inputId}>{label}</InputLabel>
        <Input
          id={inputId}
          type="number"
          defaultValue={arg.default}
          onChange={(event) => onChange(event.target.value)}
        />
      </FormControl>
    );
  } else {
    throw new Error(`Unknown type ${arg.type}`);
  }
};

export default ConstructorArgInput;
