import React, { forwardRef, Ref } from 'react';
import {
  Typography,
  TypographyPropsVariantOverrides,
  TypographyVariant,
} from '@mui/material';
import { SystemProps, TypographyProps } from '@mui/system';

export type TextProps = TypographyProps & { variant: TypographyVariant } & {
  children: React.ReactNode;
  id?: string;
};

export const Text = forwardRef((props: TextProps, ref: Ref<HTMLElement>) => {
  return (
    <Typography {...props} ref={ref}>
      {props?.children || null}
    </Typography>
  );
});

Text.displayName = 'Text';

export const LargerBoldText: React.FC<
  TypographyProps & SystemProps & { children: React.ReactNode; id?: string }
> = (props) => {
  return (
    <Text fontSize="h5.fontSize" {...props}>
      {props.children}
    </Text>
  );
};
