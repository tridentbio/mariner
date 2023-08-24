import {
  FormControl,
  FormControlLabel,
  FormHelperText,
  Input,
  InputLabel,
  MenuItem,
  Select,
  Switch,
} from '@mui/material';
import { TypeIdentifier } from '@hooks/useModelOptions';
import useModelBuilder from './hooks/useModelBuilder';

interface ArgOption {
  key: string;
  label?: string;
  latex?: string;
}
export interface ConstructorArgInputProps {
  arg: {
    type: TypeIdentifier;
    default: any;
    options?: ArgOption[];
    required?: boolean;
  };
  value: any;
  label: string;
  error?: boolean;
  helperText?: string;
  onChange: (value: any) => void;
}

const getLabel = (argOption: string | ArgOption) => {
  if (typeof argOption === 'string') {
    return argOption;
  } else {
    return argOption.label || argOption.key;
  }
};
const ConstructorArgInput = ({
  arg,
  value,
  label,
  error,
  helperText,
  onChange,
}: ConstructorArgInputProps) => {
  const formControlStyle = { minWidth: 200 };
  const inputId = `arg-option-${label}`;
  const variant = 'standard';
  const { editable } = useModelBuilder();

  if (arg.options && arg.options.length) {
    return (
      <FormControl error={error} sx={formControlStyle}>
        <InputLabel variant={variant} htmlFor="arg-option">
          {label}
        </InputLabel>
        <Select
          variant={variant}
          componentsProps={{
            root: {
              //@ts-ignore
              'data-argtype': 'options',
              id: inputId,
            },
          }}
          defaultValue={arg.default || null}
          onChange={(event) => onChange(event.target.value)}
          disabled={!editable}
        >
          {arg.options.map((option) => (
            <MenuItem
              key={option.key}
              sx={{ minWidth: '100%' }}
              value={option.key}
            >
              {getLabel(option)}
            </MenuItem>
          ))}
        </Select>
        <FormHelperText>{helperText}</FormHelperText>
      </FormControl>
    );
  } else if (arg.type === 'boolean') {
    return (
      <FormControl error={error} sx={formControlStyle}>
        <FormControlLabel
          control={
            <Switch
              inputProps={{
                //@ts-ignore
                'data-argtype': arg.type,
              }}
              id={inputId}
              defaultValue={arg.default || null}
              checked={value}
              onChange={(event) => onChange(event.target.checked)}
              disabled={!editable}
            />
          }
          label={label}
          labelPlacement="end"
        />

        <FormHelperText>{helperText}</FormHelperText>
      </FormControl>
    );
  } else if (arg.type === 'string') {
    return (
      <FormControl error={error} sx={formControlStyle}>
        <InputLabel variant={variant} htmlFor={inputId}>
          {label}
        </InputLabel>
        <Input
          inputProps={{
            'data-argtype': arg.type,
          }}
          id={inputId}
          defaultValue={arg.default || null}
          onChange={(event) => onChange(event.target.value)}
          disabled={!editable}
        />

        <FormHelperText>{helperText}</FormHelperText>
      </FormControl>
    );
  } else if (arg.type === 'number') {
    return (
      <FormControl error={error} sx={formControlStyle}>
        <InputLabel variant={variant} htmlFor={inputId}>
          {label}
        </InputLabel>
        <Input
          inputProps={{
            'data-argtype': arg.type,
          }}
          id={inputId}
          type="number"
          defaultValue={arg.default || null}
          onChange={(event) => onChange(event.target.value)}
          disabled={!editable}
        />

        <FormHelperText>{helperText}</FormHelperText>
      </FormControl>
    );
  } else {
    throw new Error(`Unknown type ${arg.type}`);
  }
};

export default ConstructorArgInput;
