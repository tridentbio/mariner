import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  SxProps,
  Typography,
  styled,
} from '@mui/material';
import { GridExpandMoreIcon } from '@mui/x-data-grid';

export const CustomAccordionStylish = styled('div')(({ theme }) => ({
  border: `1px solid ${theme.palette.divider}`,
  '&:last-of-type': {
    borderBottomRightRadius: 12,
    borderBottomLeftRadius: 12,
  },
  '&:first-of-type': {
    borderTopRightRadius: 12,
    borderTopLeftRadius: 12,
  },
  '&:not(:last-child)': {
    borderBottom: 0,
  },
  '&:before, .MuiAccordion-root:before': {
    display: 'none',
  },
  padding: 5,
}));

export const CustomAccordion = ({
  children,
  title,
  textProps,
  sx,
  testId,
  defaultExpanded,
}: {
  children: React.ReactNode;
  title: string | React.ReactNode;
  textProps?: Record<string, any>;
  sx?: SxProps;
  testId?: string;
  defaultExpanded?: boolean;
}) => {
  return (
    <CustomAccordionStylish>
      <Accordion
        disableGutters
        elevation={0}
        sx={sx}
        data-testid={`${testId}-accordion`}
        defaultExpanded={defaultExpanded}
      >
        <AccordionSummary expandIcon={<GridExpandMoreIcon />}>
          <Typography sx={textProps || {}}>{title}</Typography>
        </AccordionSummary>
        <AccordionDetails>{children}</AccordionDetails>
      </Accordion>
    </CustomAccordionStylish>
  );
};
