import { InputLabel, Switch, TextField } from '@mui/material';
import { TypeIdentifier } from './types';

interface ConstructorArgInputProps {
  arg: {
    type: TypeIdentifier;
    default: any;
  };
  value: any;
  label: string;
  onChange: (value: any) => void;
}
const ConstructorArgInput = ({
  arg,
  value,
  label,
  onChange,
}: ConstructorArgInputProps) => {
  if (arg.type === 'bool') {
    return (
      <>
        <InputLabel>{label}</InputLabel>
        <Switch
          defaultValue={arg.default}
          checked={value}
          onChange={(event) => onChange(event.target.checked)}
        />
      </>
    );
  } else if (arg.type === 'string') {
    return (
      <TextField
        defaultValue={arg.default}
        label={label}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  } else if (arg.type === 'number') {
    return (
      <TextField
        type="number"
        defaultValue={arg.default}
        label={label}
        onChange={(event) => onChange(event.target.value)}
      />
    );
  } else {
    throw new Error(`Unknown type ${arg.type}`);
  }
};

export default ConstructorArgInput;
