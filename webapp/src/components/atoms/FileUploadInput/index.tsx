import { Theme } from '@emotion/react';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import Button from '@mui/material/Button';
import { SxProps, styled } from '@mui/material/styles';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

interface InputFileUploadProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  buttonProps?: {
    startIcon?: React.ReactNode;
    sx?: SxProps<Theme>;
  };
  error?: boolean;
}

export const InputFileUpload = (props: InputFileUploadProps) => {
  return (
    <Button
      component="label"
      variant="contained"
      startIcon={props.buttonProps?.startIcon || <CloudUploadIcon />}
      sx={{
        ...props.buttonProps?.sx,
        border: props.error ? '1px solid red' : 'initial',
      }}
    >
      {props.label || 'Upload file'}
      <VisuallyHiddenInput
        type="file"
        {...props}
        className={props.error ? 'invalid' : ''}
      />
    </Button>
  );
};
