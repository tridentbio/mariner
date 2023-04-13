import { Text } from '@components/molecules/Text';
import useModelEditor from '@hooks/useModelEditor';
import { ReactNode, useState } from 'react';

export interface BaseNodeProps {
  title: string;
  children?: ReactNode;
  id: string;
  docs?: string;
  docsLink?: string;
  contentEditable?: boolean;
  handlesElement?: ReactNode;
  headerExtra?: ReactNode;
}

const BaseNode = (props: BaseNodeProps) => {
  const { expandedNodesMap } = useModelEditor();
  const displayingChildren = expandedNodesMap[props.id];

  return (
    <div>
      {props.headerExtra && <div className="header">{props.headerExtra}</div>}
      <div className="content">
        <Text fontSize={20} fontWeight="bold">
          {props.title}
        </Text>
        {props.handlesElement}
        {props.children && (
          <div
            style={{
              marginTop: 10,
              height: displayingChildren ? 'fit-content' : 0,
              transition: 'height 1s',
              display: 'flex',
              overflow: 'hidden',
              flexDirection: 'column',
            }}
          >
            {props.children}
          </div>
        )}
      </div>
    </div>
  );
};

export default BaseNode;
