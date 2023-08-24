import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  SxProps,
  Typography,
} from '@mui/material';
import { GridExpandMoreIcon } from '@mui/x-data-grid';

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
    <Accordion
      sx={sx}
      data-testid={`${testId}-accordion`}
      defaultExpanded={defaultExpanded}
    >
      <AccordionSummary expandIcon={<GridExpandMoreIcon />}>
        <Typography sx={textProps || {}}>{title}</Typography>
      </AccordionSummary>
      <AccordionDetails>{children}</AccordionDetails>
    </Accordion>
  );
};
