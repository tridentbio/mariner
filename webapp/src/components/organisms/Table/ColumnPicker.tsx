import { ChevronRight, ExpandMore } from '@mui/icons-material';
import { TreeItem, TreeView, TreeViewPropsBase } from '@mui/lab';
import {
	Box,
	Checkbox,
	Popover,
	PopoverProps,
	TextField,
	Typography,
} from '@mui/material';
import {
	ChangeEvent,
	MouseEvent,
	MouseEventHandler,
	useEffect,
	useState,
} from 'react';

interface ColumnPickerProps extends PopoverProps {
  open: boolean;
  /** in pixels */
  height?: number;
}

interface TreeNode {
  id: string;
  name: string;
  parent?: string;
  children?: TreeNode[];
}

export const ColumnPicker = (props: ColumnPickerProps) => {
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [expandedTrees, setExpandedTrees] = useState<string[]>([]);
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
  const [filteredNodes, setFilteredNodes] = useState<TreeNode[]>(data);

  //BFS algorithm to find node by his ID
  const bfsSearch = (
    graph: TreeNode[],
    targetId: any
  ): TreeNode | undefined => {
    const queue = [...graph];

    while (queue.length > 0) {
      const currNode = queue.shift();
      if (currNode?.id === targetId) {
        return currNode as TreeNode;
      } else if (currNode?.children) {
        queue.push(...currNode.children);
      }
    }
    return undefined; // Target node not found
  };

  /* Retrieve all ids from node to his children's */
  const getAllIds = (node: TreeNode, idList: string[] = []) => {
    idList.push(node.id);

    if (node.children) {
      node.children.forEach((child: any) => getAllIds(child, idList));
    }
    return idList;
  };
  /* Get IDs of all children from specific node */
  const getAllChild = (id: any) => {
    const foundNode = bfsSearch(data, id);

    return foundNode ? getAllIds(foundNode) : [];
  };

  /* Get all father IDs from specific node */
  const getAllFathers = (id: string, list: string[] = []): string[] => {
    const node = bfsSearch(data, id);

    if (node?.parent) {
      list.push(node.parent);

      return getAllFathers(node.parent, list);
    }

    return list;
  };

  const isAllChildrenChecked = (node: any, list: any) => {
    const allChild = getAllChild(node.id);
    const nodeIdIndex = allChild.indexOf(node.id);
    allChild.splice(nodeIdIndex, 1);

    return allChild.every((nodeId) =>
      selectedNodes.concat(list).includes(nodeId)
    );
  };

  const handleNodeSelect = (event: MouseEvent<HTMLElement>, nodeId: string) => {
    event.stopPropagation();

    const allChild = getAllChild(nodeId);
    const fathers = getAllFathers(nodeId);

    if (selectedNodes.includes(nodeId)) {
      // Need to de-check
      setSelectedNodes((prevSelectedNodes) =>
        prevSelectedNodes.filter((id) => !allChild.concat(fathers).includes(id))
      );
    } else {
      // Need to check
      const toBeChecked = allChild;
      for (let i = 0; i < fathers.length; ++i) {
        if (isAllChildrenChecked(bfsSearch(data, fathers[i]), toBeChecked)) {
          toBeChecked.push(fathers[i]);
        }
      }
      setSelectedNodes((prevSelectedNodes) =>
        [...prevSelectedNodes].concat(toBeChecked)
      );
    }
  };

  /* prevent the click event from propagating to the checkbox */
  const handleExpandClick: MouseEventHandler<HTMLLIElement> = (event) => {
    event.stopPropagation();
  };

  //! TO REMOVE
  useEffect(() => {
    console.log(JSON.stringify(selectedNodes, null, 4));
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
        label={
          <>
            <Checkbox
              checked={selectedNodes.indexOf(node.id) !== -1}
              tabIndex={-1}
              disableRipple
              size="small"
              onClick={(event) => handleNodeSelect(event, node.id)}
            />
            <Typography variant="caption">{node.name}</Typography>
          </>
        }
      >
        {Array.isArray(node.children)
          ? node.children.map((node) => renderTree(node))
          : null}
      </TreeItem>
    );
  };

  const expandAll = () => {
    const nodesToExpandIdList = data
      .map((node) => {
        let nodeStore: string[] = [node.id];

        return getNodesToExpandIdList(node, nodeStore);
      })
      .flat();

    setExpandedTrees(nodesToExpandIdList);
  };

  useEffect(() => {
    expandAll();
  }, []);

  //? -----------------------------------------------------------

  const nodeNameMatcher = (filterText: string, node: TreeNode) => {
    return node.name.toLowerCase().indexOf(filterText.toLowerCase()) !== -1;
  };

  const findNode = (
    node: TreeNode,
    filter: string,
    matcher: (filter: string, node: TreeNode) => boolean
  ): boolean => {
    return (
      matcher(filter, node) || //? i match
      !!(
        node.children && //? or i have decendents and one of them match
        node.children.length &&
        node.children.some((child) => findNode(child, filter, matcher))
      )
    );
  };

  const filterTreeChildrenByName = (
    node: TreeNode,
    filter: string,
    matcher = nodeNameMatcher
  ): TreeNode => {
    //? If im an exact match then all my children get to stay
    if (matcher(filter, node) || !node.children) return node;

    //? If not then only keep the ones that match or have matching descendants
    const filtered = node.children
      .filter((child) => findNode(child, filter, matcher))
      .map((child) => filterTreeChildrenByName(child, filter, matcher));

    return Object.assign({}, node, { children: filtered });
  };

  const expandFilteredNodeChildren = (
    node: TreeNode,
    filter: string,
    matcher = nodeNameMatcher
  ): TreeNode => {
    let children = node.children;

    if (!children || children.length === 0) {
      return Object.assign({}, node, { toggled: false });
    }

    const childrenWithMatches = (node.children || []).filter((child) =>
      findNode(child, filter, matcher)
    );

    const shouldExpand = childrenWithMatches.length > 0;

    // If im going to expand, go through all the matches and see if thier children need to expand
    if (shouldExpand) {
      children = childrenWithMatches.map((child) => {
        return expandFilteredNodeChildren(child, filter, matcher);
      });
    }

    return Object.assign({}, node, {
      chiNode: children,
      toggled: shouldExpand,
    });
  };

  const nodeHasChildren = (node: TreeNode) =>
    node.children && node.children.length > 0;

  const getNodesToExpandIdList = (node: TreeNode, store: string[]) => {
    if (!nodeHasChildren(node)) {
      return store;
    }

    if (node.children) {
      node.children.forEach(
        (child) => nodeHasChildren(child) && store.push(child.id)
      );
      node.children.forEach((child) => getNodesToExpandIdList(child, store));
    }

    return store;
  };

  /**
   * Find tree item with recursive approach
   */
  const searchTree = (node: TreeNode, nodeId: string): TreeNode | null => {
    if (node.id === nodeId) return node;

    if (node.children != null) {
      let foundNode: TreeNode | null = null;

      for (
        let index = 0;
        foundNode == null && index < node.children.length;
        index++
      ) {
        foundNode = searchTree(node.children[index], nodeId);
      }

      return foundNode;
    }

    return null;
  };

  const onColumnFilterChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const value = e.target.value;
    const filter = value.trim();

    if (!filter) {
      setFilteredNodes(data);
      setExpandedTrees([]);
      return;
    }

    let filteredNodes: TreeNode[] = [];
    let store: string[] = [];

    data.forEach((node) => {
      const filteredChidrenNodes = filterTreeChildrenByName(node, filter);

      if (
        filteredChidrenNodes.children?.length ||
        nodeNameMatcher(filter, node)
      ) {
        filteredNodes.push(
          expandFilteredNodeChildren(filteredChidrenNodes, filter)
        );

        if (filteredChidrenNodes.children?.length) {
          let nodeStore: string[] = [node.id];

          getNodesToExpandIdList(filteredChidrenNodes, nodeStore);

          store.push(...nodeStore);
        }
      }
    });

    setExpandedTrees(store);
    setFilteredNodes(filteredNodes);
  };

  const handleToggle: TreeViewPropsBase['onNodeToggle'] = (
    event,
    nodeIdList
  ) => {
    setExpandedTrees(nodeIdList);
  };

  const onPopoverClose: PopoverProps['onClose'] = (event, reason) => {
    setFilteredNodes(data);

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
            onChange={(e) => onColumnFilterChange(e)}
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
          expanded={expandedTrees}
          selected={selectedNodes}
          multiSelect
          defaultCollapseIcon={<ExpandMore />}
          defaultExpandIcon={<ChevronRight />}
          onNodeToggle={handleToggle}
          sx={{
            overflowY: 'scroll',
            padding: 1,
            height: props.height ? props.height - 100 : 300,
          }}
        >
          {filteredNodes.map((node) => renderTree(node))}
        </TreeView>
      </Box>
    </Popover>
  );
};
