import { ReactNode } from 'react';

const NotFound = (props: { children: ReactNode }) => {
  return <div style={{ fontWeight: 'bold' }}>{props.children}</div>;
};

export default NotFound;
