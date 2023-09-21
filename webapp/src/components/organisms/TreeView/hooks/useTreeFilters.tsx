import { ChangeEvent, useState } from 'react';

export const useTreeFilters = ({ treeView }: { treeView: TreeNode[] }) => {
  const [filteredNodes, setFilteredNodes] = useState<TreeNode[]>(treeView);

  const nodeNameMatcher = (filterText: string, node: TreeNode) => {
    return node.name.toLowerCase().indexOf(filterText.toLowerCase()) !== -1;
  };

  const nodeExistsInHierarchy = (
    node: TreeNode,
    filter: string,
    matcher: (filter: string, node: TreeNode) => boolean
  ): boolean => {
    return (
      matcher(filter, node) || //? i match
      !!(
        node.children && //? or i have decendents and one of them match
        node.children.length &&
        node.children.some((child) =>
          nodeExistsInHierarchy(child, filter, matcher)
        )
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
      .filter((child) => nodeExistsInHierarchy(child, filter, matcher))
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
      nodeExistsInHierarchy(child, filter, matcher)
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

  const getTreesToExpandIdList = (node: TreeNode, store: string[]) => {
    if (!nodeHasChildren(node)) {
      return store;
    }

    if (node.children) {
      node.children.forEach(
        (child) => nodeHasChildren(child) && store.push(child.id)
      );
      node.children.forEach((child) => getTreesToExpandIdList(child, store));
    }

    return store;
  };

  const onColumnFilterChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
    treesToExpandCallback: (value: string[]) => void
  ) => {
    const value = e.target.value;
    const filter = value.trim();

    if (!filter) {
      setFilteredNodes(treeView);
      treesToExpandCallback([]);
      return;
    }

    let filteredNodes: TreeNode[] = [];
    let store: string[] = [];

    treeView.forEach((node) => {
      const filteredChidren = filterTreeChildrenByName(node, filter);

      if (filteredChidren.children?.length || nodeNameMatcher(filter, node)) {
        filteredNodes.push(expandFilteredNodeChildren(filteredChidren, filter));

        if (filteredChidren.children?.length) {
          let nodeStore: string[] = [node.id];

          getTreesToExpandIdList(filteredChidren, nodeStore);

          store.push(...nodeStore);
        }
      }
    });

    treesToExpandCallback(store);
    setFilteredNodes(filteredNodes);
  };

  const resetFilters = () => setFilteredNodes(treeView);

  return {
    filteredNodes,
    onColumnFilterChange,
    getTreesToExpandIdList,
    resetFilters,
    nodeExistsInHierarchy,
  };
};
