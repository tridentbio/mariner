import { Text } from '@components/molecules/Text';
import { ReactNode } from 'react';

const NoData = ({ children }: { children?: ReactNode }) => (
  <Text>{children || 'NO DATA'}</Text>
);

export default NoData;
