import { Chip, ChipProps, ChipTypeMap, Typography } from '@mui/material';
import { Box } from '@mui/system';
import { reprDataType } from '@utils';
import { ColumnConfig } from 'app/rtk/generated/models';
import { DataTypeGuard } from 'app/types/domain/datasets';

const DataTypeChip = ({
  sx,
  prefix,
  color,
  ...props
}: ColumnConfig['dataType'] & {
  sx?: ChipProps['sx'];
  prefix?: string;
  color?: ChipTypeMap['props']['color'];
}) => {
  const content = prefix
    ? `${prefix} - (${reprDataType(props)})`
    : reprDataType(props);
  return (
    <Chip
      sx={sx}
      color={color || 'default'}
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
            <Typography component={'span'}>{content}</Typography>
          )}
          {DataTypeGuard.isCategorical(props) && (
            <Typography ml={2} variant="overline" component={'span'}>
              NUM CLASSES: {Object.keys(props.classes).length}
            </Typography>
          )}
        </Box>
      }
    />
  );
};

export default DataTypeChip;
