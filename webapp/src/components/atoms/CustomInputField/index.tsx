import { TextField, TextFieldProps } from '@mui/material';
import React, {
  ChangeEvent,
  InputHTMLAttributes,
  useEffect,
  useState,
} from 'react';

interface CustomInputFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  validate?: (value: string | number) => boolean;
}

const CustomInputField: React.FC<TextFieldProps & CustomInputFieldProps> = ({
  onChange,
  onBlur,
  validate,
  value: incomingValue,
  ...props
}) => {
  const [value, setValue] = useState<string | number>(
    incomingValue as string | number
  );
  const [color, setTextColor] = useState<string>();

  const handleChangeValue = (e: ChangeEvent<HTMLInputElement>) => {
    onChange && onChange(e);
    let newValue: number | string = e.target.value;
    if (props.type === 'number') {
      newValue = e.target.value;
    }
    setValue(newValue.toString());
  };

  useEffect(() => {
    if (value !== incomingValue) {
      setValue(incomingValue as string | number);
    }
  }, [incomingValue]);

  useEffect(() => {
    if (validate) {
      const isValid = validate(value);
      setTextColor(isValid ? 'green' : 'red');
    }
  }, [value]);
  return (
    <TextField
      {...props}
      onChange={handleChangeValue}
      onBlur={onBlur}
      value={value}
      inputProps={{
        style: { color },
        //? Avoid dragging when input is focused (causing continuous increment/decrement when using input stepper)
        className: 'nodrag',
        ...(props.inputProps || {})
      }}
    />
    //TODO: Fix bug with leading 0 that never goes away, probably a MUI bug
  );
};

export default CustomInputField;
