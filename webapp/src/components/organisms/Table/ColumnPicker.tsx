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

interface ColumnPickerProps {
  open: boolean;
  /** in pixels */
  height?: number;
  treeView: TreeNode[];
  onChange?: (selectedColumns: string[]) => void;
  defaultSelectedColumns?: string[];
  popoverProps?: Omit<PopoverProps, 'onChange' | 'open'>;
}

export const ColumnPicker = (props: ColumnPickerProps) => {
  const [expandedTrees, setExpandedTrees] = useState<string[]>([]);

  const {
    filteredNodes,
    getTreesToExpandIdList,
    onColumnFilterChange,
    resetFilters,
    nodeExistsInHierarchy,
  } = useTreeFilters({ treeView: props.treeView });

  const expandAll = () => {
    const nodesToExpandIdList = props.treeView
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

    props.popoverProps?.onClose && props.popoverProps.onClose(event, reason);
  };

  const handleNodesSelect: TreeViewProps['onSelect'] = (selectedNodes) => {
    //? Discard nodes that are not related directly to the column (e.g. nodes with children/expansible)
    const columnNodes = selectedNodes.filter((nodeId) => {
      return props.treeView.some((node) => {
        return nodeExistsInHierarchy(
          node,
          nodeId,
          (filter, n) => n.id === filter && !n.children
        );
      });
    });

    props.onChange && props.onChange(columnNodes);
  };

  return (
    <Popover
      {...props.popoverProps}
      open={props.open}
      sx={{
        ...props.popoverProps?.sx,
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
          treeView={props.treeView}
          filteredTreeView={filteredNodes}
          expanded={expandedTrees}
          multiSelect
          onSelect={handleNodesSelect}
          defaultSelectedNodes={props.defaultSelectedColumns}
          onNodeToggle={handleToggle}
          sx={{
            overflowY: 'scroll',
            padding: 1,
            height: props.height ? props.height - 100 : 300,
          }}
          renderTreeItemLabel={(node, params) => {
            const isCurrentNodeChecked =
              params.selectedNodes.indexOf(node.id) !== -1;
            const isOnlyOneChecked = params.selectedNodes.length == 1;

            return (
              <>
                <Checkbox
                  checked={isCurrentNodeChecked}
                  tabIndex={-1}
                  disableRipple
                  size="small"
                  disabled={isOnlyOneChecked && isCurrentNodeChecked}
                  onClick={(event) => params.handleNodeSelect(event, node.id)}
                />
                <Typography variant="caption">{node.name}</Typography>
              </>
            );
          }}
        />
      </Box>
    </Popover>
  );
};
