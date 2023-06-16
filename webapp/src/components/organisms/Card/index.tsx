import { Card as MuiCard, CardContent, SxProps } from '@mui/material';
import { Theme } from '@mui/system';
import { LargerBoldText } from 'components/molecules/Text';
import { FC, ReactNode } from 'react';

interface CardProps {
  children?: ReactNode;
  title?: ReactNode | string;
  sx?: SxProps<Theme>;
}

const CardTitle = ({ title }: { title: string | ReactNode }) => {
  if (!title) return null;
  else if (typeof title === 'string')
    return <LargerBoldText>{title}</LargerBoldText>;
  return <>{title}</>;
};

const Card: FC<CardProps> = ({ sx, children, title }) => {
  return (
    <MuiCard sx={sx} variant="outlined">
      <CardContent>
        <CardTitle title={title} />
        {children}
      </CardContent>
    </MuiCard>
  );
};

export default Card;
