import { useGetModelOptionsQuery } from 'app/rtk/generated/models';
import { positionNodes } from './utils';
import { ApplySuggestionsCommandArgs } from 'model-compiler/src/implementation/commands/ApplySuggestionsCommand';
import { DeleteCommandArgs } from 'model-compiler/src/implementation/commands/DeleteComponentsCommand';
import { EditComponentsCommandArgs } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import TorchModelEditorImpl from 'model-compiler/src/implementation/TorchModelEditorImpl';
import { SchemaContextTypeGuard } from 'model-compiler/src/implementation/SchemaContext';
import Suggestion from 'model-compiler/src/implementation/Suggestion';
import {
  ComponentConfigType,
  LayerFeaturizerType,
  NodeType,
  ModelSchema,
  NodePositionTypesMap,
} from '@model-compiler/src/interfaces/torch-model-editor';
import { iterateTopologically, unwrapDollar } from 'model-compiler/src/utils';
import { createContext, useContext, useMemo, ReactNode, useState } from 'react';
import {
  Edge,
  useReactFlow,
  useNodesState,
  useEdgesState,
  useStore,
  useNodesInitialized,
} from 'reactflow';
import { ArrayElement, isArray, Required } from 'utils';
import { ITorchModelEditorContext, MarinerNode } from './types';
import {
  getSourceHandles,
  getTargetHandles,
} from 'model-compiler/src/implementation/modelSchemaQuery';

// Context impement `ITorchModelEditorContext`
// @ts-ignore
const TorchModelEditorContext = createContext<ITorchModelEditorContext>({});

/**
 * A provider for the ITorchModelEditorContext.
 *
 * Each provider should wrap a single TorchModelEditorV2 component
 */
export const TorchModelEditorContextProvider = ({
  children,
}: {
  children?: ReactNode;
}) => {
  const {
    addComponent,
    editComponent,
    deleteComponents,
    getSuggestions,
    applySuggestions,
  } = new TorchModelEditorImpl();
  const reactFlowValue = useReactFlow<NodeType, any>();
  const [schema, setSchema] = useState<ModelSchema | undefined>();
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const nodeInternals = useStore((state) => {
    return state.nodeInternals;
  });
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [expandedNodesMap, setExpandedNodesMap] = useState<
    Record<string, boolean>
  >({});
  const { data: modelOptions } = useGetModelOptionsQuery();
  const nodesInitialized = useNodesInitialized();

  const optionsByLib: ITorchModelEditorContext['optionsByLib'] = useMemo(() => {
    return (
      modelOptions?.reduce((acc, model) => {
        const lib = model.classPath.split('.')[0];
        if (!acc[lib]) {
          acc[lib] = [];
        }
        acc[lib].push(model);
        return acc;
      }, {} as Record<string, typeof modelOptions>) || {}
    );
  }, [modelOptions]);

  const optionsByType: ITorchModelEditorContext['options'] = useMemo(() => {
    return modelOptions?.reduce((acc, option) => {
      acc[option.classPath] = option;
      return acc;
    }, {} as Record<string, ArrayElement<typeof modelOptions>>);
  }, [modelOptions]);

  const expandNodes: ITorchModelEditorContext['expandNodes'] = (
    nodeIds?: string[]
  ) => {
    const newNodesExpanded = !nodeIds
      ? nodes.map((node) => node.id)
      : Object.keys(expandedNodesMap).concat(nodeIds);
    const newMap = newNodesExpanded.reduce((acc, node) => {
      acc[node] = true;
      return acc;
    }, {} as Record<string, boolean>);
    setExpandedNodesMap(newMap);
  };

  const contractNodes: ITorchModelEditorContext['contractNodes'] = (
    nodeIds?: string[]
  ) => {
    if (!nodeIds) {
      setExpandedNodesMap({});
      return;
    }
    const newExpadedNodeIds = Object.keys(expandedNodesMap).filter(
      (nodeId) => !nodeIds.includes(nodeId)
    );
    const newMap = newExpadedNodeIds.reduce((acc, node) => {
      acc[node] = true;
      return acc;
    }, {} as Record<string, boolean>);
    setExpandedNodesMap(newMap);
  };

  const suggestionsByNode: ITorchModelEditorContext['suggestionsByNode'] =
    useMemo(() => {
      return suggestions.reduce((acc, cur) => {
        if (SchemaContextTypeGuard.isNodeSchema(cur.context)) {
          if (cur.context.nodeId in acc)
            acc[cur.context.nodeId] = acc[cur.context.nodeId].concat([cur]);
          else acc[cur.context.nodeId] = [cur];
        }
        return acc;
      }, {} as Record<string, Suggestion[]>);
    }, [suggestions]);

  const suggestionsByEdge: ITorchModelEditorContext['suggestionsByEdge'] =
    useMemo(() => {
      return suggestions.reduce((acc, cur) => {
        if (SchemaContextTypeGuard.isEdgeSchema(cur.context)) {
          if (cur.context.edgeId in acc)
            acc[cur.context.edgeId] = acc[cur.context.edgeId].concat([cur]);
          else acc[cur.context.edgeId] = [cur];
        }
        return acc;
      }, {} as Record<string, Suggestion[]>);
    }, [suggestions]);

  /**
   * Gets the default value of a python type
   *
   * @param {string} type - string containing python primitive
   */
  const getDefaultType = (type: string) => {
    if (type.includes('int')) {
      return 0;
    } else if (type.includes('float')) {
      return 0.0;
    } else if (type.includes('bool')) {
      return false;
    } else if (type.includes('str')) {
      return '';
    }
    return undefined;
  };
  /**
   * Creates the default node data for the component's type (classPath)
   *
   * @template T - component classPath string literal
   * @param {T} type - classPath
   * @param {Required<typeof modelOptions>} options - component options
   * @returns {ComponentConfigType<T>} default node data of component with type `classPath`
   */
  const makeDefaultData = <T extends LayerFeaturizerType['type']>(
    type: T,
    options: Required<typeof modelOptions>
  ): ComponentConfigType<T> => {
    const opt = options.find(
      (opt) =>
        (opt.component.type as Required<LayerFeaturizerType['type']>) === type
    );
    if (!opt) {
      // eslint-disable-next-line no-console
      console.error('Faile to find summary data on layer ' + type);
      // @ts-ignore
      return { forwardArgs: {}, constructorArgs: {} };
    }
    // @ts-ignore
    return {
      type,
      forwardArgs: Object.keys(opt.component.forwardArgsSummary || {}).reduce(
        (acc, key) => {
          acc[key] = '';
          return acc;
        },
        {} as Record<string, string>
      ),
      constructorArgs: Object.keys(
        opt.component.constructorArgsSummary || {}
      ).reduce((acc, key) => {
        // @ts-ignore
        const type = opt.component.constructorArgsSummary[key];
        // @ts-ignore
        acc[key] = opt.defaultArgs[key] || getDefaultType(type);
        return acc;
      }, {} as Record<string, any>),
    };
  };

  /**
   * Gets a map between node ids and it's positions
   */
  const getNodePositionsMap = (): Record<string, { x: number; y: number }> => {
    return nodes.reduce((acc, cur) => {
      acc[cur.id] = cur.position;
      return acc;
    }, {} as Record<string, { x: number; y: number }>);
  };

  const schemaToEditorGraph: ITorchModelEditorContext['schemaToEditorGraph'] = (
    schema: ModelSchema,
    positions?: ReturnType<typeof getNodePositionsMap>
  ): [MarinerNode[], Edge[]] => {
    const nodes: MarinerNode[] = [];
    let edges: Edge[] = [];
    const map = positions ? positions : getNodePositionsMap();
    // @ts-ignore
    iterateTopologically(schema, (node, type) => {
      nodes.push({
        id: node.name,
        data: node,
        type,
        position: map[node.name] || {
          x: Math.random() * 400,
          y: Math.random() * 400,
        },
      });
      if (node.type === 'input') return;

      Object.entries(node.forwardArgs || {})
        .map(
          ([key, dep]) =>
            [key, isArray(dep) ? dep.map(unwrapDollar) : unwrapDollar(dep)] as [
              string,
              string | string[]
            ]
        )
        .forEach(([key, dep]) => {
          if (!dep) return;
          if (isArray(dep)) {
            edges = edges.concat(
              dep
                .filter((el) => !!el)
                .map((depElement) => {
                  const [head, ...tail] = depElement.split('.');
                  return {
                    type: 'ModelEdge',
                    id: `${node.name}-${depElement}`,
                    source: head,
                    sourceHandle: tail.join('.'),
                    target: node.name,
                    targetHandle: key,
                  };
                })
            );
          } else {
            const [head, ...tail] = dep.split('.');
            edges.push({
              type: 'ModelEdge',
              id: `${node.name}-${dep}`,
              source: head,
              sourceHandle: tail.join('.'),
              target: node.name,
              targetHandle: key,
            });
          }
        });
    });
    return [nodes, edges];
  };

  const applyDagreLayout: ITorchModelEditorContext['applyDagreLayout'] = (
    direction,
    spaceValue,
    _edges
  ) => {
    setNodes((nodes) =>
      positionNodes(nodes, _edges || edges, direction, spaceValue)
    );
  };

  const getMiddlePosition = (positions: Position[]) => {
    const x = positions.reduce((acc, cur) => acc + cur.x, 0) / positions.length;
    const y = positions.reduce((acc, cur) => acc + cur.y, 0) / positions.length;

    return { x, y };
  };

  const convertRelativePositionsToAbsolute = (
    nodePositionMap: NodePositionTypesMap
  ) => {
    if (!schema || !Object.keys(nodePositionMap).length) return undefined;

    const positionsMap = getNodePositionsMap();

    Object.entries(nodePositionMap).forEach(([nodeName, nodePosition]) => {
      switch (nodePosition.type) {
        case 'relative': {
          const referenceNodesNameList: string[] = nodePosition.references;

          const referenceNodesPositions: Position[] =
            referenceNodesNameList.reduce<Position[]>(
              (acc, referenceNodeName) => {
                if (referenceNodeName in positionsMap)
                  acc.push(positionsMap[referenceNodeName]);
                return acc;
              },
              []
            );

          positionsMap[nodeName] = getMiddlePosition(referenceNodesPositions);
          break;
        }
        default:
          positionsMap[nodeName] = { x: nodePosition.x, y: nodePosition.y };
      }
    });

    return positionsMap;
  };

  /**
   * Updates react flow context state based on new schema
   */
  const updateNodesAndEdges = (
    schema: ModelSchema,
    positionsMap?: ReturnType<typeof getNodePositionsMap>
  ) => {
    if (!positionsMap) positionsMap = getNodePositionsMap();
    setSchema(schema);
    // @ts-ignore
    const [nodes, edges] = schemaToEditorGraph(schema, positionsMap);
    setNodes(nodes);
    setEdges(edges);
    // @ts-ignore
    setSuggestions(getSuggestions({ schema }).getSuggestions());
  };

  const addComponentWithPosition: ITorchModelEditorContext['addComponent'] = (
    args,
    position
  ) => {
    if (!modelOptions) throw 'too early';
    const map = getNodePositionsMap();
    map[args.name] = position;
    const schema = addComponent({
      schema: args.schema,
      type: args.componentType,
      // @ts-ignore
      data: {
        ...(makeDefaultData(args.type, modelOptions) as object),
        name: args.name,
      },
    });
    updateNodesAndEdges(schema, map);
    return schema;
  };

  const editComponentAndApply: ITorchModelEditorContext['editComponent'] = (
    args: EditComponentsCommandArgs
  ) => {
    const schema = editComponent(args);
    updateNodesAndEdges(schema);
    return schema;
  };

  const deleteComponentAndApply: ITorchModelEditorContext['deleteComponents'] =
    (args: DeleteCommandArgs) => {
      const schema = deleteComponents(args);
      updateNodesAndEdges(schema);
      return schema;
    };

  const applySuggestionsWithPositions = (args: ApplySuggestionsCommandArgs) => {
    const { schema, updatedNodePositions } = applySuggestions(args);

    const updatedPositions =
      convertRelativePositionsToAbsolute(updatedNodePositions);

    updateNodesAndEdges(schema, updatedPositions);

    return schema;
  };

  const toggleExpanded: ITorchModelEditorContext['toggleExpanded'] = (
    nodeId: string
  ) => {
    if (nodeId in expandedNodesMap) contractNodes([nodeId]);
    else expandNodes([nodeId]);
  };

  const getSuggestionsAndSet: ITorchModelEditorContext['getSuggestions'] = (
    args
  ) => {
    const info = getSuggestions(args);
    setSuggestions(info.getSuggestions());
    return info;
  };

  type Position = { x: number; y: number };
  const length = ({ x: xa, y: ya }: Position, { x: xb, y: yb }: Position) => {
    const [dx, dy] = [xb - xa, yb - ya];
    return Math.sqrt(dx * dx + dy * dy);
  };
  const char = (index: number): string =>
    String.fromCharCode('a'.charCodeAt(0) + index);

  const byDistance =
    (_nodeInternals: typeof nodeInternals, point: { x: number; y: number }) =>
    (a: { nodeId: string }, b: { nodeId: string }): number => {
      const aInternal = _nodeInternals.get(a.nodeId);
      const bInternal = _nodeInternals.get(b.nodeId);
      if (aInternal === bInternal) return 0;
      if (!aInternal || (bInternal?.position && !aInternal?.position)) return 1;
      else if (!bInternal || (aInternal?.position && !bInternal?.position))
        return -1;
      else
        return (
          length(aInternal.position, point) - length(bInternal.position, point)
        );
    };
  const assignPositionsOrdering: ITorchModelEditorContext['assignPositionsOrdering'] =
    (type, component) => {
      if (!schema) return [];
      if (!modelOptions) return [];
      const pressedPosition = reactFlowValue.getNode(component.name)
        ?.position || { x: 3, y: 3 };
      const distanceMeasure = byDistance(nodeInternals, pressedPosition);
      const isNotConnected = (handle: { isConnected?: boolean }) =>
        !handle.isConnected;
      const isNotFromComponent =
        (c: typeof component) => (handle: { nodeId: string }) =>
          handle.nodeId !== c.name;

      if (type === 'source') {
        // @ts-ignore
        const handles = getTargetHandles(schema);
        const result = handles
          .sort(distanceMeasure)
          .filter(isNotFromComponent(component))
          .filter(isNotConnected)
          .map((handle, index) => ({
            key: char(index),
            nodeId: handle.nodeId,
            targetHandleId: handle.handle,
          }));
        setHandleKeysByNodeId(result);
        return result;
      } else if (type === 'target') {
        // @ts-ignore
        const result = getSourceHandles(schema, modelOptions)
          .sort(distanceMeasure)
          .filter(isNotFromComponent(component))
          .filter(isNotConnected)
          .map((handle, index) => ({
            key: char(index),
            nodeId: handle.nodeId,
            sourceHandleId: handle.handle,
          }));
        setHandleKeysByNodeId(result);
        return result;
      }
      throw new Error('type must be "source" or "target"');
    };
  const [handleKeysByNodeId, setHandleKeysByNodeId] = useState<
    ReturnType<ITorchModelEditorContext['assignPositionsOrdering']> | undefined
  >();
  const getHandleKey: ITorchModelEditorContext['getHandleKey'] = (
    nodeId,
    type,
    handleid
  ) => {
    if (!handleKeysByNodeId) return;
    const node = handleKeysByNodeId.find((item) => {
      if (item.nodeId !== nodeId) return false;
      if ('sourceHandleId' in item) {
        return (
          (!handleid && !item.sourceHandleId) ||
          item.sourceHandleId === handleid
        );
      } else if ('targetHandleId' in item) {
        return (
          (!handleid && !item.targetHandleId) ||
          item.targetHandleId === handleid
        );
      }
      return false;
    });
    return node?.key;
  };
  const clearPositionOrdering: ITorchModelEditorContext['clearPositionOrdering'] =
    () => {
      setHandleKeysByNodeId([]);
    };
  return (
    <TorchModelEditorContext.Provider
      value={{
        ...reactFlowValue,
        nodes,
        edges,
        setNodes,
        setEdges,
        options: optionsByType,
        optionsByLib,
        addComponent: addComponentWithPosition,
        editComponent: editComponentAndApply,
        deleteComponents: deleteComponentAndApply,
        getSuggestions: getSuggestionsAndSet,
        applySuggestions: applySuggestionsWithPositions,
        schemaToEditorGraph,
        applyDagreLayout,
        expandNodes,
        contractNodes,
        toggleExpanded,
        expandedNodesMap,
        schema,
        setSchema,
        suggestions,
        suggestionsByNode,
        suggestionsByEdge,
        assignPositionsOrdering,
        getHandleKey,
        keyAssignments: handleKeysByNodeId,
        clearPositionOrdering,
        onNodesChange,
        onEdgesChange,
        nodesInitialized,
      }}
    >
      {children}
    </TorchModelEditorContext.Provider>
  );
};

/**
 * Hook to get value of the `TorchModelEditorContextProvider` that parents the caller
 * component
 */
const useTorchModelEditor = (): ITorchModelEditorContext => {
  const value = useContext(TorchModelEditorContext);
  return { ...value };
};

export default useTorchModelEditor;
