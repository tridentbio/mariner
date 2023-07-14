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
}: {
  children: React.ReactNode;
  title: string;
  textProps?: Record<string, any>;
  sx?: SxProps;
}) => {
  return (
    <Accordion sx={sx}>
      <AccordionSummary expandIcon={<GridExpandMoreIcon />}>
        <Typography sx={textProps}>{title}</Typography>
      </AccordionSummary>
      <AccordionDetails>{children}</AccordionDetails>
    </Accordion>
  );
};
