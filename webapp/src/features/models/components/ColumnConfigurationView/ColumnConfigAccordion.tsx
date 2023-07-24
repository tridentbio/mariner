import { ColumnConfig } from '@app/rtk/generated/models';
import DataTypeChip from '@components/atoms/DataTypeChip';
import { CustomAccordion } from '@components/molecules/CustomAccordion';
interface ColumnConfigurationAccordionProps {
  name: string;
  dataType: ColumnConfig['dataType'];
  textProps?: Record<string, any>;
  children: React.ReactNode;
}

const ColumnConfigurationAccordion = ({
  name,
  dataType,
  textProps = {},
  children,
}: ColumnConfigurationAccordionProps) => {
  return (
    <CustomAccordion
      title={<DataTypeChip prefix={name} {...dataType} />}
      textProps={textProps}
    >
      {children}
    </CustomAccordion>
  );
};

export default ColumnConfigurationAccordion;
