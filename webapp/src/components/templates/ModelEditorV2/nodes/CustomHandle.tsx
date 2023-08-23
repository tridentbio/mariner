import { Typography } from '@mui/material';
import useModelEditor from 'hooks/useModelEditor';
import { Handle, HandleProps } from 'reactflow';

export interface CustomHandleProps extends HandleProps {
  required?: boolean;
  order: number;
  total: number;
  nodeId: string;
}

const total = '250px';
const CustomHandle = (props: CustomHandleProps) => {
  const { total: slots, order, nodeId, required, ...handleProps } = props;
  const { getHandleKey } = useModelEditor();
  const key = getHandleKey(nodeId, handleProps.type, handleProps.id || '');
  return (
    <Handle
      {...handleProps}
      style={{
        height: 10,
        width: 10,
        backgroundColor: required ? 'black' : 'lightgrey',
        left: `calc(${total} * ${(order + 1) / (slots + 1)} )`,
      }}
    >
      <Typography
        sx={{
          top: order % 2 == 0 ? '-20px' : '8px',
          position: 'relative',
          fontFamily: 'monospace',
          fontSize: 10,
        }}
      >
        {handleProps.id}
      </Typography>
      {key && (
        <div
          style={{
            backgroundColor: '#2e332f',
            padding: 10,
            top: '-80px',
            position: 'relative',
            color: 'white',
            borderRadius: 3,
            textAlign: 'center',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {key}
        </div>
      )}
    </Handle>
  );
};

export default CustomHandle;
