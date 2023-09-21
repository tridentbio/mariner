import { ChevronRight, ExpandMore } from '@mui/icons-material';
import {
  TreeView as MuiTreeView,
  MultiSelectTreeViewProps,
  TreeItem,
} from '@mui/lab';
import { isArray } from '@utils';
import { useSelectedNodes } from './hooks/useSelectedNodes';
import { useEffect } from 'react';

export interface TreeViewProps
  extends Omit<MultiSelectTreeViewProps, 'selected' | 'children' | 'onSelect'> {
  treeView: TreeNode[];
  filteredTreeView?: TreeNode[];
  renderTreeItemLabel: (
    node: TreeNode,
    params: Pick<
      ReturnType<typeof useSelectedNodes>,
      'selectedNodes' | 'handleNodeSelect'
    >
  ) => JSX.Element;
  /** node ID list */
  onSelect?: (selectedNodes: string[]) => void;
  defaultSelectedNodes?: string[];
}

export const TreeView = ({
  treeView,
  filteredTreeView,
  expanded,
  onNodeToggle,
  sx,
  renderTreeItemLabel,
  onSelect,
  defaultSelectedNodes,
}: TreeViewProps) => {
  const { selectedNodes, handleNodeSelect, handleExpandClick } =
    useSelectedNodes({ treeView, defaultSelectedNodes });

  useEffect(() => {
    onSelect && onSelect(selectedNodes);
  }, [selectedNodes]);

  const renderTree = (node: TreeNode) => {
    return (
      <TreeItem
        key={node.id}
        nodeId={node.id}
        onClick={handleExpandClick}
        sx={{
          '& .MuiTreeItem-content': {
            width: 'initial',
          },
        }}
        label={renderTreeItemLabel(node, { selectedNodes, handleNodeSelect })}
      >
        {Array.isArray(node.children)
          ? node.children.map((node) => renderTree(node))
          : null}
      </TreeItem>
    );
  };

  return (
    <MuiTreeView
      expanded={expanded}
      selected={selectedNodes}
      multiSelect
      defaultCollapseIcon={<ExpandMore />}
      defaultExpandIcon={<ChevronRight />}
      onNodeToggle={(...event) => onNodeToggle && onNodeToggle(...event)}
      sx={sx}
    >
      {(isArray(filteredTreeView) ? filteredTreeView : treeView).map((node) =>
        renderTree(node)
      )}
    </MuiTreeView>
  );
};
