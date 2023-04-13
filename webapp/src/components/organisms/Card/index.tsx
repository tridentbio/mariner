import { Card as MuiCard, CardContent, SxProps } from '@mui/material';
import { Theme } from '@mui/system';
import { LargerBoldText } from 'components/molecules/Text';
import { FC, ReactNode } from 'react';

interface CardProps {
  children?: ReactNode;
  title?: ReactNode;
  sx?: SxProps<Theme>;
}

const Card: FC<CardProps> = ({ sx, children, title }) => {
  return (
    <MuiCard sx={sx} variant="outlined">
      <CardContent>
        {title && <LargerBoldText>{title}</LargerBoldText>}
        {children}
      </CardContent>
    </MuiCard>
  );
};

export default Card;
