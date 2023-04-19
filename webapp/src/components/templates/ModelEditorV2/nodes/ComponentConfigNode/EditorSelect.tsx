import { MenuItem, TextField } from '@mui/material';
import { ComponentOption } from 'app/rtk/generated/models';
import { useEffect, useRef, useState } from 'react';

interface EditorSelectProps {
  editable?: boolean;
  errors: any;
  editConstrutorArgs: (key: string, value: any) => void;
  option: ComponentOption;
  argKey: string;
  value: any;
}

const EditorSelect = (props: EditorSelectProps) => {
  const [open, setOpen] = useState(false);

  const { editable, errors, editConstrutorArgs, option } = props;

  const handleOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const selectRef = useRef<HTMLSelectElement>(null);
  if (!option?.argsOptions || !(props.argKey in option.argsOptions)) {
    console.warn('Rendering a select without options');

    return null;
  }

  useEffect(() => {
    selectRef &&
      selectRef.current &&
      editable &&
      selectRef.current.addEventListener('mousedown', () => {
        setOpen((opened) => !opened);
      });
  }, []);

  return (
    <TextField
      sx={{ mb: 2, width: '100%' }}
      key={props.argKey}
      value={props.value || ''}
      onChange={(event) => {
        editConstrutorArgs(props.argKey, event.target.value);
      }}
      error={props.argKey in errors}
      label={errors[props.argKey] || props.argKey}
      disabled={!editable}
      select
      SelectProps={{
        open,
        onOpen: handleOpen,
        onClose: handleClose,
        ref: selectRef,
        sx: {
          mb: 2,
          width: '80%',
          maxHeight: 60,
          fontSize: 16,
          py: -5,
          color: 'grey.600',
          zIndex: 100,
        },
      }}
    >
      {option.argsOptions[props.argKey].map((item) => (
        <MenuItem key={item} value={item}>
          {item}
        </MenuItem>
      ))}
    </TextField>
  );
};

export default EditorSelect;
