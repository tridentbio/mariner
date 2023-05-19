import { Box } from '@mui/material';
import { Text } from './Text';

interface SectionProps extends Record<string, any> {
  title: string;
  children: React.ReactNode;
}

export const Section = ({ children, title, ...rest }: SectionProps) => {
  return (
    <Box sx={{ mb: 1 }}>
      <Text fontWeight="bold">{title}:</Text>
      <Box {...rest}>{children}</Box>
    </Box>
  );
};
