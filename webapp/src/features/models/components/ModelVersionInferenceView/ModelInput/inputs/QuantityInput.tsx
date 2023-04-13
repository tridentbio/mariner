import { TextField, Box } from '@mui/material';
import { Text } from 'components/molecules/Text';

interface QuantityInputProps {
  value: number;
  onChange: (val: number) => any;
  unit: string;
  label: string;
}

const QuantityInput = ({
  label,
  unit,
  value,
  onChange,
}: QuantityInputProps) => {
  return (
    <Box
      sx={{
        width: 300,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
      }}
    >
      <TextField
        label={label}
        sx={{ mr: 3 }}
        type="number"
        value={value}
        onChange={(event) => onChange(parseFloat(event.target.value))}
      />
      <Text>{unit}</Text>
    </Box>
  );
};

export default QuantityInput;
