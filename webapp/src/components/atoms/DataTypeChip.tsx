import { Chip, ChipProps, Typography } from '@mui/material';
import { Box } from '@mui/system';
import { reprDataType } from '@utils';
import { ColumnConfig } from 'app/rtk/generated/models';
import { DataTypeGuard } from 'app/types/domain/datasets';

const DataTypeChip = ({
  sx,
  prefix,
  ...props
}: ColumnConfig['dataType'] & { sx?: ChipProps['sx']; prefix?: string }) => {
  const content = prefix
    ? `${prefix} - (${reprDataType(props)})`
    : reprDataType(props);
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
          {props.domainKind && <Typography>{content}</Typography>}
          {DataTypeGuard.isCategorical(props) && (
            <Typography ml={2} variant="overline">
              NUM CLASSES: {Object.keys(props.classes).length}
            </Typography>
          )}
        </Box>
      }
    />
  );
};

export default DataTypeChip;
