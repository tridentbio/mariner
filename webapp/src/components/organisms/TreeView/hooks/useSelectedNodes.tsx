import { MouseEvent, MouseEventHandler, useEffect, useState } from 'react';

export const useSelectedNodes = ({
  treeView,
  defaultSelectedNodes,
}: {
  treeView: TreeNode[];
  defaultSelectedNodes?: string[];
}) => {
  const [selectedNodes, setSelectedNodes] = useState<string[]>(
    defaultSelectedNodes || []
  );

  useEffect(() => {
    handleDefaultSelectedNodes();
  }, []);

  const handleDefaultSelectedNodes = () => {
    if (defaultSelectedNodes?.length) {
      let checkedNodeIdList: string[] = [];

      defaultSelectedNodes?.forEach((nodeId) => {
        const allChild = getAllChild(nodeId);
        const fathers = getAllFathers(nodeId);

        const toBeChecked = allChild.filter(
          (child) => !selectedNodes.includes(child)
        );

        for (let i = 0; i < fathers.length; ++i) {
          const foundNode = searchTree(treeView, fathers[i]);

          if (
            foundNode &&
            isAllChildrenChecked(foundNode, toBeChecked) &&
            !checkedNodeIdList.includes(fathers[i])
          ) {
            toBeChecked.push(fathers[i]);
          }
        }

        checkedNodeIdList = checkedNodeIdList.concat(toBeChecked);
      });

      setSelectedNodes((prev) => [...prev].concat(checkedNodeIdList));
    }
  };

  //BFS algorithm to find node by his ID
  const searchTree = (
    treeView: TreeNode[],
    targetNodeId: string
  ): TreeNode | undefined => {
    const queue = [...treeView];

    while (queue.length > 0) {
      const currNode = queue.shift();
      if (currNode?.id === targetNodeId) {
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
      node.children.forEach((child) => getAllIds(child, idList));
    }
    return idList;
  };

  /* Get IDs of all children from specific node */
  const getAllChild = (id: string) => {
    const foundNode = searchTree(treeView, id);

    return foundNode ? getAllIds(foundNode) : [];
  };

  /* Get all father IDs from specific node */
  const getAllFathers = (id: string, list: string[] = []): string[] => {
    const node = searchTree(treeView, id);

    if (node?.parent) {
      list.push(node.parent);

      return getAllFathers(node.parent, list);
    }

    return list;
  };

  const isAllChildrenChecked = (node: TreeNode, list: string[]) => {
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
      // Need to uncheck
      setSelectedNodes((prevSelectedNodes) =>
        prevSelectedNodes.filter((id) => !allChild.concat(fathers).includes(id))
      );
    } else {
      // Need to check
      const toBeChecked = allChild.filter(
        (child) => !selectedNodes.includes(child)
      );

      for (let i = 0; i < fathers.length; ++i) {
        const foundNode = searchTree(treeView, fathers[i]);

        if (foundNode && isAllChildrenChecked(foundNode, toBeChecked)) {
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

  return {
    selectedNodes,
    handleNodeSelect,
    handleExpandClick,
  };
};
