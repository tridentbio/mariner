import { ColumnConfig } from '@app/rtk/generated/models';
import DataTypeChip from '@components/atoms/DataTypeChip';
import { CustomAccordion } from '@components/molecules/CustomAccordion';
interface ColumnConfigurationAccordionProps {
  name: string;
  dataType: ColumnConfig['dataType'];
  textProps?: Record<string, any>;
  children: React.ReactNode;
  testId?: string;
}

const ColumnConfigurationAccordion = ({
  name,
  dataType,
  textProps = {},
  children,
  testId,
}: ColumnConfigurationAccordionProps) => {
  return (
    <CustomAccordion
      testId={testId}
      title={<DataTypeChip prefix={name} {...dataType} />}
      textProps={textProps}
    >
      {children}
    </CustomAccordion>
  );
};

export default ColumnConfigurationAccordion;
