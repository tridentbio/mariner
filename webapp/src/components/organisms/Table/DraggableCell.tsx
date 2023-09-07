import { Column } from '@components/templates/Table';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { TableCell } from '@mui/material';

export const DraggableCell = ({
  col,
  id,
  children,
}: {
  id: string;
  col: Column<any, any>;
  children: JSX.Element;
}) => {
  const {
    attributes,
    listeners,
    transform,
    transition,
    setNodeRef,
    isDragging,
  } = useSortable({ id: id, disabled: col.fixed });

  const style = {
    transform: CSS.Translate.toString(transform),
    transition,
  };

  return (
    <TableCell
      sx={{
        ...(col.customSx || {}),
        cursor: col.fixed ? 'initial' : isDragging ? 'grabbing' : 'grab',
        ...(isDragging && !col.fixed
          ? {
              backgroundColor: '#e0e0e03d',
              borderRadius: '7px',
            }
          : {}),
      }}
      style={style}
      ref={setNodeRef}
      {...attributes}
      {...listeners}
    >
      {children}
    </TableCell>
  );
};
