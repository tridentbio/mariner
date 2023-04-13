import { useState } from 'react';
import {
  RadioGroup,
  FormControlLabel,
  Radio,
  Box,
  FormControl,
  FormLabel,
  TextField,
} from '@mui/material';
import JSME from './JSME';

interface SmileInputProps {
  value: string;
  onChange: (value: string) => any;
  label: string;
}

type EditorOption = 'jsme' | 'input';

const SmileInput = ({ value, onChange, label }: SmileInputProps) => {
  // return <JSME width={'300px'} height={'300px'} />;
  const initialValue = 'jsme';
  const [editor, setEditor] = useState<EditorOption>(initialValue);

  return (
    <Box>
      <FormControl sx={{ display: 'flex', flexDirection: 'column' }}>
        <FormLabel id="demo-radio-buttons-group-label">{label}</FormLabel>
        <RadioGroup
          sx={{ display: 'flex', flexDirection: 'row' }}
          value={editor}
          aria-labelledby="demo-radio-buttons-group-label"
          defaultValue={initialValue}
          name="radio-buttons-group"
          onChange={(event) => setEditor(event.target.value as EditorOption)}
        >
          <FormControlLabel
            value="jsme"
            control={<Radio />}
            label="JSME Input"
          />
          <FormControlLabel
            value="input"
            control={<Radio />}
            label="Textbox Input"
          />
        </RadioGroup>
      </FormControl>
      {editor === 'input' ? (
        <TextField
          data-testid="input-textbox"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          label={label}
        />
      ) : (
        <JSME
          width={'500px'}
          height={'300px'}
          smiles={value}
          onChange={onChange}
        />
      )}
    </Box>
  );
};

export default SmileInput;
