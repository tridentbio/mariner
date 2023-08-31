import {
  Box,
  Checkbox,
  Popover,
  PopoverProps,
  TextField,
  Typography,
} from '@mui/material';
import { useEffect, useState } from 'react';
import { TreeView, TreeViewProps } from '../TreeView';
import { useTreeFilters } from '../TreeView/hooks/useTreeFilters';

interface ColumnPickerProps extends PopoverProps {
  open: boolean;
  /** in pixels */
  height?: number;
}

export const ColumnPicker = (props: ColumnPickerProps) => {
  const [data, setData] = useState<TreeNode[]>([
    {
      id: '1',
      name: 'Parent 1',
      children: [
        {
          id: '2',
          name: 'Child 1',
          parent: '1',
          children: [
            {
              id: '5',
              name: 'Grandchild 1',
              parent: '2',
              children: [
                {
                  id: '9',
                  name: 'Great-grandchild 1',
                  parent: '5',
                },
                {
                  id: '10',
                  name: 'Great-grandchild 2',
                  parent: '5',
                },
              ],
            },
            {
              id: '6',
              name: 'Grandchild 2',
              parent: '2',
              children: [
                {
                  id: '11',
                  name: 'Great-grandchild 3',
                  parent: '6',
                },
                {
                  id: '12',
                  name: 'Great-grandchild 4',
                  parent: '6',
                },
              ],
            },
          ],
        },
        {
          id: '3',
          name: 'Child 2',
          parent: '1',
          children: [
            {
              id: '7',
              name: 'Grandchild x',
              parent: '3',
            },
          ],
        },
      ],
    },
    {
      id: '4',
      name: 'Parent 2',
      children: [
        {
          id: '8',
          name: 'Child 3',
          parent: '4',
          children: [
            {
              id: '13',
              name: 'Grandchild 4',
              parent: '8',
              children: [
                {
                  id: '14',
                  name: 'Great-grandchild 5',
                  parent: '13',
                },
                {
                  id: '15',
                  name: 'Great-grandchild 6',
                  parent: '13',
                },
              ],
            },
            {
              id: '16',
              name: 'Grandchild 5',
              parent: '8',
              children: [
                {
                  id: '17',
                  name: 'Great-grandchild 7',
                  parent: '16',
                },
                {
                  id: '18',
                  name: 'Great-grandchild y',
                  parent: '16',
                },
              ],
            },
          ],
        },
      ],
    },
  ]);
  const [expandedTrees, setExpandedTrees] = useState<string[]>([]);
  const {
    filteredNodes,
    getTreesToExpandIdList,
    onColumnFilterChange,
    resetFilters,
  } = useTreeFilters({ treeView: data });

  const expandAll = () => {
    const nodesToExpandIdList = data
      .map((node) => {
        let nodeStore: string[] = [node.id];

        return getTreesToExpandIdList(node, nodeStore);
      })
      .flat();

    setExpandedTrees(nodesToExpandIdList);
  };

  useEffect(() => {
    expandAll();
  }, []);

  const handleToggle: TreeViewProps['onNodeToggle'] = (event, nodeIdList) => {
    setExpandedTrees(nodeIdList);
  };

  const onPopoverClose: PopoverProps['onClose'] = (event, reason) => {
    resetFilters();

    props.onClose && props.onClose(event, reason);
  };

  return (
    <Popover
      {...props}
      sx={{
        ...props.sx,
        '& .MuiPopover-paper': {
          borderRadius: 2,
        },
      }}
      onClose={onPopoverClose}
    >
      <Box>
        <Box sx={{ padding: '0.7rem 0.5rem' }}>
          <TextField
            label="Search"
            variant="outlined"
            onChange={(e) => {
              onColumnFilterChange(e, (treesToExpand) => {
                if (!treesToExpand.length) expandAll();
                else setExpandedTrees(treesToExpand);
              });
            }}
            fullWidth
            size="small"
            inputProps={{
              style: { fontSize: '1rem' },
            }}
            InputLabelProps={{
              sx: { fontSize: '1rem' },
            }}
          />
        </Box>
        <TreeView
          treeView={data}
          filteredTreeView={filteredNodes}
          expanded={expandedTrees}
          multiSelect
          onNodeToggle={handleToggle}
          sx={{
            overflowY: 'scroll',
            padding: 1,
            height: props.height ? props.height - 100 : 300,
          }}
          renderTreeItemLabel={(node, params) => (
            <>
              <Checkbox
                checked={params.selectedNodes.indexOf(node.id) !== -1}
                tabIndex={-1}
                disableRipple
                size="small"
                onClick={(event) => params.handleNodeSelect(event, node.id)}
              />
              <Typography variant="caption">{node.name}</Typography>
            </>
          )}
        />
      </Box>
    </Popover>
  );
};
