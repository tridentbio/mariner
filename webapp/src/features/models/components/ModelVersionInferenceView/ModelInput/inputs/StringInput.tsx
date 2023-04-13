import { TextField } from '@mui/material';

interface StringInputProps {
  onChange: (value: string) => any;
  value: string;
  label: string;
}
const StringInput = ({ label, onChange, value }: StringInputProps) => {
  return (
    <TextField
      sx={{ width: 300 }}
      onChange={(event) => onChange(event.target.value)}
      value={value}
      label={label}
    />
  );
};

export default StringInput;
