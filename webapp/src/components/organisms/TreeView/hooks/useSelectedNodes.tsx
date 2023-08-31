import { MouseEvent, MouseEventHandler, useState } from 'react';

export const useSelectedNodes = ({ treeView }: { treeView: TreeNode[] }) => {
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);

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
    const foundNode = bfsSearch(treeView, id);

    return foundNode ? getAllIds(foundNode) : [];
  };

  /* Get all father IDs from specific node */
  const getAllFathers = (id: string, list: string[] = []): string[] => {
    const node = bfsSearch(treeView, id);

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
        if (
          isAllChildrenChecked(bfsSearch(treeView, fathers[i]), toBeChecked)
        ) {
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
