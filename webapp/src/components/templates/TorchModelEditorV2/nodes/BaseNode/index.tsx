import { Text } from 'components/molecules/Text';
import useTorchModelEditor from 'hooks/useTorchModelEditor';
import { ReactNode } from 'react';
import styled from 'styled-components';

export interface BaseNodeProps {
  title: string;
  children?: ReactNode;
  id: string;
  docs?: string;
  docsLink?: string;
  contentEditable?: boolean;
  handlesElement?: ReactNode;
  headerExtra?: ReactNode;
  selected?: boolean;
}

const BaseNodeContainer = styled.div<{
  selected?: boolean;
}>`
  ${(props) =>
    props.selected
      ? `
      outline: 1px solid transparent;
      background: 
          linear-gradient(90deg, #333 50%, transparent 0) repeat-x,
          linear-gradient(90deg, #333 50%, transparent 0) repeat-x,
          linear-gradient(0deg, #333 50%, transparent 0) repeat-y,
          linear-gradient(0deg, #333 50%, transparent 0) repeat-y;
      background-size: 12px 1px, 12px 1px, 1px 12px, 1px 12px;
      background-position: 0 0, 0 100%, 0 0, 100% 0;
      animation: linearGradientMove .5s infinite linear;
    `
      : ''}

  @keyframes linearGradientMove {
    100% {
      background-position: 12px 0, -12px 100%, 0 -12px, 100% 12px;
    }
  }
`;

const BaseNode = (props: BaseNodeProps) => {
  const { expandedNodesMap } = useTorchModelEditor();
  const displayingChildren = expandedNodesMap[props.id];

  return (
    <BaseNodeContainer selected={props.selected}>
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
              padding: displayingChildren ? 10 : 0,
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
    </BaseNodeContainer>
  );
};

export default BaseNode;
