// Move to molecules
import { FormHelperText, InputLabel } from '@mui/material';
import { Box } from '@mui/system';
import MDEditor, { MDEditorProps } from '@uiw/react-md-editor';
import { FocusEventHandler } from 'react';
import rehypeSanitize from 'rehype-sanitize';

interface MDTextFieldProps extends Omit<MDEditorProps, 'onChange'> {
  value?: string;
  defaultValue?: string;
  onChange: (mdStr: string) => any;
  error?: boolean;
  helperText?: string;
  label?: string;
  onBlur?: FocusEventHandler<any>;
}

const MDTextField = ({
  error,
  label,
  value,
  onChange,
  onBlur,
  helperText,
  defaultValue,
  ...editorProps
}: MDTextFieldProps) => {
  return (
    <Box data-color-mode="light" sx={{ mb: 1 }}>
      <InputLabel htmlFor="description-input" error={error}>
        {label}
      </InputLabel>
      <MDEditor
        previewOptions={{ rehypePlugins: [[rehypeSanitize]] }}
        value={value}
        defaultValue={defaultValue}
        id="description-input"
        onChange={(mdStr) => onChange(mdStr || '')}
        onBlur={onBlur}
        {...editorProps}
      />
      {error && (
        <FormHelperText error={error} variant="outlined">
          {helperText}
        </FormHelperText>
      )}
    </Box>
  );
};

export default MDTextField;
