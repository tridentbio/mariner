import { InputLabel, MenuItem, Switch, TextField } from '@mui/material';
import { TypeIdentifier } from '@hooks/useModelOptions';

export interface ConstructorArgInputProps {
  arg: {
    type: TypeIdentifier;
    default: any;
    options?: any[];
    required?: boolean;
  };
  value: any;
  label: string;
  error?: boolean;
  helpText?: string;
  onChange: (value: any) => void;
}
const ConstructorArgInput = ({
  arg,
  value,
  label,
  error,
  helpText,
  onChange,
}: ConstructorArgInputProps) => {
  if (arg.type === 'boolean') {
    return (
      <>
        {/* //? Color change workaround (Input label `color` prop doesn't seem to be working properly) */}
        <InputLabel sx={{ color: error ? '#d32f2f' : null }}>
          {label}
        </InputLabel>
        <Switch
          defaultValue={arg.default || null}
          checked={!!value}
          onChange={(event) => onChange(event.target.checked)}
        />
      </>
    );
  } else if (arg.type === 'string' && arg.options) {
    return (
      <TextField
        select
        error={error}
        helperText={helpText}
        defaultValue={arg.default || null}
        label={label}
        onChange={(event) => onChange(event.target.value)}
      >
        {arg.options.map((option) => (
          <MenuItem key={option} sx={{ width: '100%' }} value={option}>
            {option}
          </MenuItem>
        ))}
      </TextField>
    );
  } else if (arg.type === 'string') {
    return (
      <TextField
        defaultValue={arg.default || null}
        error={error}
        helperText={helpText}
        label={label}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  } else if (arg.type === 'number') {
    return (
      <TextField
        type="number"
        error={error}
        helperText={helpText}
        defaultValue={arg.default || null}
        label={label}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  } else {
    throw new Error(`Unknown type ${arg.type}`);
  }
};

export default ConstructorArgInput;
