import { ColumnConfig } from '@app/rtk/generated/models';
import DataTypeChip from '@components/atoms/DataTypeChip';
import { CustomAccordion } from '@components/molecules/CustomAccordion';
import { ChipTypeMap } from '@mui/material';
interface ColumnConfigurationAccordionProps {
  name: string;
  dataType: ColumnConfig['dataType'];
  textProps?: Record<string, any>;
  children: React.ReactNode;
  testId?: string;
  defaultExpanded?: boolean;
  labelColor?: ChipTypeMap['props']['color'];
}

const ColumnConfigurationAccordion = ({
  name,
  dataType,
  textProps = {},
  children,
  testId,
  defaultExpanded,
  labelColor,
}: ColumnConfigurationAccordionProps) => {
  return (
    <CustomAccordion
      testId={testId}
      title={<DataTypeChip prefix={name} {...dataType} color={labelColor} />}
      textProps={textProps}
      defaultExpanded={defaultExpanded}
    >
      {children}
    </CustomAccordion>
  );
};

export default ColumnConfigurationAccordion;
