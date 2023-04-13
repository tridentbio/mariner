import { Chip, ChipProps, Typography } from '@mui/material';
import { Box } from '@mui/system';
import { ColumnConfig } from '@app/rtk/generated/models';
import { DataTypeGuard } from '@app/types/domain/datasets';
import { fixDomainKindCasing } from '@hooks/useModelEditor/utils';

const DataTypeChip = ({
  sx,
  ...props
}: ColumnConfig['dataType'] & { sx?: ChipProps['sx'] }) => {
  return (
    <Chip
      sx={sx}
      label={
        <Box
          sx={{
            padding: 1,
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'flex-start',
          }}
        >
          {props.domainKind && (
            <Typography>{fixDomainKindCasing(props.domainKind)}</Typography>
          )}
          {DataTypeGuard.isCategorical(props) && (
            <Typography ml={2} variant="overline">
              NUM CLASSES: {Object.keys(props.classes).length}
            </Typography>
          )}
          {DataTypeGuard.isQuantity(props) && (
            <Typography ml={2} variant="overline">
              {props.unit}
            </Typography>
          )}
        </Box>
      }
    />
  );
};

export default DataTypeChip;
