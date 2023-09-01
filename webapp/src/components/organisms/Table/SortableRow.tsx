import { Column } from '@components/templates/Table';
import {
  DndContext,
  DragEndEvent,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  closestCorners,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  SortableContext,
  horizontalListSortingStrategy,
} from '@dnd-kit/sortable';
import { TableRow, TableRowProps } from '@mui/material';
import { useMemo } from 'react';

export const SortableRow = ({
  children,
  columns,
  onDropped,
}: TableRowProps & {
  columns: Column<any, any>[];
  onDropped?: (columns: Column<any, any>[]) => void;
}) => {
  const columnIdList = useMemo<string[]>(
    () => columns.map((col, index) => index.toString()),
    [columns]
  );

  const sensors = useSensors(
    useSensor(MouseSensor, {}),
    useSensor(TouchSensor, {}),
    useSensor(KeyboardSensor, {})
  );

  const reorder = (startIndex: number, endIndex: number) => {
    const result = Array.from(columns);
    const [removed] = result.splice(startIndex, 1);
    result.splice(endIndex, 0, removed);

    return result;
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const columnOverRef = columns[Number(over.id)];

      if (columnOverRef.disableDraggable) return;

      const reordeneredColumns = reorder(Number(active.id), Number(over.id));

      onDropped && onDropped(reordeneredColumns);
    }
  };

  return (
    <DndContext
      sensors={sensors}
      onDragEnd={handleDragEnd}
      collisionDetection={closestCorners}
    >
      <SortableContext
        items={columnIdList}
        strategy={horizontalListSortingStrategy}
      >
        <TableRow>{children}</TableRow>
      </SortableContext>
    </DndContext>
  );
};
