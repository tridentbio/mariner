import { ColumnConfig } from '@app/rtk/generated/models';
import DataTypeChip from '@components/atoms/DataTypeChip';
import { CustomAccordion } from '@components/molecules/CustomAccordion';
interface ColumnConfigurationAccordionProps {
  name: string;
  dataType: ColumnConfig['dataType'];
  textProps?: Record<string, any>;
  children: React.ReactNode;
  testId?: string;
  defaultExpanded?: boolean;
}

const ColumnConfigurationAccordion = ({
  name,
  dataType,
  textProps = {},
  children,
  testId,
  defaultExpanded,
}: ColumnConfigurationAccordionProps) => {
  return (
    <CustomAccordion
      testId={testId}
      title={<DataTypeChip prefix={name} {...dataType} />}
      textProps={textProps}
      defaultExpanded={defaultExpanded}
      sx={{
        padding: 0.5,
        boxShadow: '0px 0px 5px rgba(0, 0, 0, 0.1)',
        // borderRadius: 3,
      }}
    >
      {children}
    </CustomAccordion>
  );
};

export default ColumnConfigurationAccordion;
