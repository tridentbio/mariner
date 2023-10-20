import {
  ModelTemplate
} from '@app/rtk/generated/models';
import { Text } from '@components/molecules/Text';
import { AppTabsProps } from '@components/organisms/Tabs';
import { useTorchModelEditor } from '@hooks';
import { MarinerNode } from '@hooks/useTorchModelEditor/types';
import {
  FeaturizersType,
  LayersType,
  ModelSchema,
  NodeType,
  TorchModelSchema
} from '@model-compiler/src/interfaces/torch-model-editor';
import { Box } from '@mui/material';
import FullScreenWrapper from 'components/organisms/FullScreenWrapper';
import TorchModelEditorControls from 'components/templates/TorchModelEditor/Components/TorchModelEditorControls';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import {
  getComponent,
  getNode,
} from 'model-compiler/src/implementation/modelSchemaQuery';
import {
  DragEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Edge,
  NodeProps,
  SelectionMode,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { deepClone, isArray, substrAfterLast } from 'utils';
import { ModelTemplates } from './ModelTemplates';
import OptionsSidebarV2 from './OptionsSidebarV2';
import { SidebarBase } from './SidebarBase';
import { SidebarToggle } from './SidebarToggle';
import SuggestionsList from './SuggestionsList';
import ModelEdge from './edges/ModelEdge';
import ComponentConfigNode from './nodes/ComponentConfigNode';
import InputNode from './nodes/InputNode';
import OutputNode from './nodes/OutputNode';
import { StyleOverrides } from './styles';

export type ModelEditorElementsCount = {
  components: number;
  templates: number;
};

type TorchModelEditorProps = {
  value: ModelSchema;
  onChange?: (schema: ModelSchema) => void;
  editable?: boolean;
  dagre?: number | 'goodDistance';
  initialElementsCount?: ModelEditorElementsCount;
  onElementsCountChange?: (count: ModelEditorElementsCount) => void;
};

type DraggingContexts = 'ComponentOption' | 'Template';

export interface EditorDragStartParams<T extends object> {
  event: DragEvent<HTMLDivElement>;
  data: T;
  type: DraggingContexts;
}

const getGoodDistance = (nodesNumber: number) => {
  if (nodesNumber < 3) return 8;
  else if (nodesNumber < 5) return 6;
  else if (nodesNumber < 7) return 3;
  else return 1;
};

const TorchModelEditor = ({
  value,
  onChange,
  editable = true,
  dagre,
  initialElementsCount,
  onElementsCountChange,
}: TorchModelEditorProps) => {
  const {
    applyDagreLayout,
    editComponent,
    addComponent,
    project,
    nodes,
    edges,
    setNodes,
    setEdges,
    schemaToEditorGraph,
    schema,
    setSchema,
    suggestions,
    getSuggestions,
    assignPositionsOrdering,
    keyAssignments,
    clearPositionOrdering,
    options,
    fitView,
    onNodesChange,
    onEdgesChange,
    nodesInitialized,
    updateNodesAndEdges,
    getNodePositionsMap,
    unselectNodesAndEdges,
    isInputNode,
    onNodesDelete,
  } = useTorchModelEditor();
  const [fullScreen, setFullScreen] = useState(false);
  const [connectingNode, setConnectingNode] = useState<
    | {
        component: NodeType;
        handleId: string | null;
        handleType: 'source' | 'target';
      }
    | undefined
  >();

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [currentSidebarTab, setCurrentSidebarTab] = useState(0);

  const handleSidebarTabChange = (tabValue: number) => {
    setCurrentSidebarTab(tabValue);
  };

  const nodeTypes = useMemo(
    () => ({
      layer: (props: any) => (
        <ComponentConfigNode editable={editable} {...props} />
      ),
      featurizer: (props: any) => (
        <ComponentConfigNode editable={editable} {...props} />
      ),
      input: (props: NodeProps) => <InputNode {...props} />,
      output: (props: NodeProps) => (
        <OutputNode {...props} editable={editable} />
      ),
    }),
    [editable]
  );
  const edgeTypes = useMemo(
    () => ({
      ModelEdge: (props: any) => <ModelEdge editable={editable} {...props} />,
    }),
    [editable]
  );

  const reactFlowWrapper = useRef<HTMLDivElement | null>(null);

  const [droppedElementsCount, setDroppedElementsCount] =
    useState<ModelEditorElementsCount>({
      components: 0,
      templates: 0,
    });

  useEffect(() => {
    if (value) {
      setSchema(value);
    }

    if (initialElementsCount) setDroppedElementsCount(initialElementsCount);
  }, []);

  useEffect(() => {
    fitView();
  }, [nodesInitialized]);

  useEffect(() => {
    onElementsCountChange && onElementsCountChange(droppedElementsCount);
  }, [droppedElementsCount]);

  useEffect(() => {
    if (!reactFlowWrapper.current) return;
    const [nodes, edges] = schemaToEditorGraph(value);

    //? Avoids <ComponentConfigNode /> edges unrender when the data for the node handles creation are not loaded yet
    if (options) {
      if (editable) {
        setNodes(nodes.reverse());
        setEdges(edges);
        getSuggestions({ schema: value });
        dagre == 'goodDistance'
          ? applyDagreLayout('TB', getGoodDistance(nodes.length))
          : applyDagreLayout('TB', 3, edges);
      } else {
        setNodes(nodes);
        setEdges(edges);
        getSuggestions({ schema: value });
        applyDagreLayout('TB', 3, edges);
      }
    }
  }, [options]);

  useEffect(() => {
    onChange && schema && onChange(schema);
  }, [schema]);

  const [draggingElement, setDraggingNode] = useState<{
    x: number;
    y: number;
    data: { [key: string]: any };
    type: DraggingContexts;
  }>();

  const onDragStart = <T extends object>({
    event,
    data,
    type,
  }: EditorDragStartParams<T>) => {
    setDraggingNode({
      data,
      type,
      x: event.clientX - event.currentTarget.getBoundingClientRect().x,
      y: event.clientY - event.currentTarget.getBoundingClientRect().y,
    });

    event.dataTransfer.effectAllowed = 'move';
  };

  /**
   * DragOver event handler for when an option from the options sidebar
   * is dragged over the canvas.
   */
  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  };

  /**
   * Drop event handler, with the side effect of adding the node with empty
   * arguments in the canvas state
   */
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    if (!reactFlowWrapper.current || !draggingElement) return;

    unselectNodesAndEdges();

    switch (draggingElement.type) {
      case 'ComponentOption':
        handleAddOption(event);
        break;
      case 'Template':
        handleAddTemplate(event);
        break;
    }

    setDraggingNode(undefined);
  };

  const handleAddOption = (event: DragEvent<HTMLDivElement>) => {
    if (!reactFlowWrapper.current || !draggingElement) return;

    const componentData = draggingElement.data;
    const { type: classPath } = componentData.component;

    if (
      componentData.type === 'scikit_reg' ||
      componentData.type === 'scikit_class' ||
      !classPath
    )
      return;

    const { type: componentType } = componentData;
    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();

    const position = project({
      x: event.clientX - reactFlowBounds.left - draggingElement.x,
      y: event.clientY - reactFlowBounds.top - draggingElement.y,
    });

    addComponent(
      {
        schema: schema as TorchModelSchema,
        name: `${substrAfterLast(classPath, '.')}-${
          droppedElementsCount.components
        }`,
        type: classPath,
        componentType,
      },
      position
    );

    setDroppedElementsCount((data) => ({
      ...data,
      components: data.components + 1,
    }));
  };

  const handleAddTemplate = (event: DragEvent<HTMLDivElement>) => {
    if (!reactFlowWrapper.current || !draggingElement) return;

    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();

    const position = project({
      x: event.clientX - reactFlowBounds.left - draggingElement.x,
      y: event.clientY - reactFlowBounds.top - draggingElement.y,
    });

    const template = draggingElement.data as ModelTemplate;

    if (schema) {
      let sourceLayer: LayersType | FeaturizersType | undefined;

      const generateTemplateNodeId = (nodeId: string) =>
        `${nodeId}-template-${droppedElementsCount.templates}`;

      //? Update layer nodes ID's and remove edges pointing to input nodes
      const templateLayers = (
        template.version.config as TorchModelSchema
      ).spec.layers?.map((layer) => {
        const layerCopy = deepClone(layer) as typeof layer;

        if ('forwardArgs' in layerCopy) {
          Object.keys(layerCopy.forwardArgs).forEach((arg) => {
            const forwardArgs = layerCopy.forwardArgs as {
              [key: string]: string | string[];
            };

            if (isArray(forwardArgs[arg])) {
              forwardArgs[arg] = (forwardArgs[arg] as string[])
                .filter((arg) => !isInputNode(arg))
                .map(generateTemplateNodeId);

              if (forwardArgs[arg].length == 0) delete forwardArgs[arg];
            } else if (isInputNode(forwardArgs[arg] as string)) {
              delete forwardArgs[arg];
            } else {
              forwardArgs[arg] = generateTemplateNodeId(
                forwardArgs[arg] as string
              );
            }

            if (!forwardArgs[arg]) sourceLayer = layerCopy;
          });
        }

        layerCopy.name = `${layerCopy.name}-template-${droppedElementsCount.templates}`;

        return layerCopy;
      });

      const updatedSchema: ModelSchema = {
        ...schema,
        spec: {
          layers: [...(schema.spec.layers || []), ...(templateLayers || [])],
        },
      };

      let templateLayerNodes: string[] = [];

      const updatedElements: {
        nodes: MarinerNode[];
        edges: Edge[];
      } = {
        nodes: [],
        edges: [],
      };

      updateNodesAndEdges(
        updatedSchema,
        !!sourceLayer
          ? {
              [sourceLayer.name]: position,
              ...getNodePositionsMap(),
            }
          : undefined,
        (nodes, edges) => {
          updatedElements.nodes = nodes;
          updatedElements.edges = edges;

          templateLayerNodes = nodes
            .filter((node) =>
              templateLayers?.some((layer) => layer.name == node.id)
            )
            .map((node) => node.id);

          return {
            nodes: nodes.map((node) => ({
              ...node,
              selected: templateLayerNodes.includes(node.id),
            })),
            edges,
          };
        }
      );

      // TODO: Fix the wrong template position coordinates not being applied on drop
      //? Dagre package does not support layout positioning
      //? Solutions: Manually update the selected nodes positions based on delta?
      applyDagreLayout(
        'TB',
        getGoodDistance(updatedElements.nodes.length),
        updatedElements.edges,
        templateLayerNodes
      );

      setDroppedElementsCount((data) => ({
        ...data,
        templates: data.templates + 1,
      }));
    }
  };

  const [listeningForKeyPress, setListeningForKeyPress] = useState(false);

  const handleKeyPress = (event: KeyboardEvent) => {
    if (!listeningForKeyPress || !keyAssignments) return;
    const key = keyAssignments.find(({ key }) => key === event.key);
    if (!key) return;
    const node = nodes.find((node) => node.id === key.nodeId)?.data;
    if (!node) return;
    if (!options) return;
    if (!schema) return;
    let source: NodeType,
      sourceHandle: string,
      target: NodeType,
      targetHandle: string;
    if (connectingNode?.handleType === 'source') {
      source = connectingNode.component;
      sourceHandle = connectingNode.handleId || '';
      target = node;
      targetHandle = 'targetHandleId' in key ? key.targetHandleId : '';
    } else if (connectingNode?.handleType === 'target') {
      target = connectingNode.component;
      targetHandle = connectingNode.handleId || '';
      source = node;
      sourceHandle = 'sourceHandleId' in key ? key.sourceHandleId : '';
    } else return;
    if (target.type === 'input') return;
    const data = makeComponentEdit({
      component: target,
      forwardArgs: {
        sourceComponentName: source.name,
        sourceComponentOutput: sourceHandle,
        targetNodeForwardArg: targetHandle,
      },
      options,
    });
    editComponent(
      {
        data,
      },
      schema
    );
  };

  useLayoutEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [listeningForKeyPress, keyAssignments]);

  const tabs: AppTabsProps['tabs'] = [
    {
      label: 'Options',
      panel: <OptionsSidebarV2 onDragStart={onDragStart} />,
    },
    {
      label: 'Templates',
      panel: <ModelTemplates onDragStart={onDragStart} />,
    },
  ];

  return (
    <FullScreenWrapper fullScreen={fullScreen} setFullScreen={setFullScreen}>
      <StyleOverrides>
        <Box
          ref={reactFlowWrapper}
          style={{
            flex: 'grow',
            left: 0,
            height: fullScreen ? '100vh' : '80vh',
            border: '1px solid black',
            overflow: 'hidden',
            position: 'relative',
          }}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodesDelete={onNodesDelete}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            minZoom={0.1}
            selectionMode={SelectionMode.Partial}
            onDragOver={onDragOver}
            onDrop={onDrop}
            onConnectEnd={() => {
              clearPositionOrdering();
            }}
            onNodeClick={(event, clickedNode) => {
              setNodes((prev) =>
                prev.map((node) => ({
                  ...node,
                  selected: node.id === clickedNode.id,
                }))
              );
            }}
            onConnectStart={(event, connectionParams) => {
              if (!schema) return;
              if (!connectionParams.handleType) {
                // eslint-disable-next-line no-console
                return console.warn(
                  'No "handleId" in connectionParams of onConnectStartEvent'
                );
              }
              if (!connectionParams.nodeId) {
                // eslint-disable-next-line no-console
                return console.warn(
                  'No "nodeId" in connectionParams of onConnectStartEvent'
                );
              }
              const component = getNode(schema, connectionParams.nodeId);
              if (!component) {
                // eslint-disable-next-line no-console
                return console.warn(
                  `No component named ${connectionParams.nodeId} in schema`
                );
              }
              setConnectingNode({
                component,
                handleId: connectionParams.handleId,
                handleType: connectionParams.handleType,
              });
              if (!component) return;
              assignPositionsOrdering(connectionParams.handleType, component);
              setListeningForKeyPress(true);
            }}
            onConnect={(connection) => {
              if (
                !connection.source ||
                !connection.target ||
                !options ||
                !schema
              )
                return;
              const component = getComponent(schema, connection.target);
              if (!component) return;
              const data = makeComponentEdit({
                component,
                forwardArgs: {
                  targetNodeForwardArg: connection.targetHandle || '',
                  sourceComponentName: connection.source,
                  sourceComponentOutput: connection.sourceHandle || undefined,
                },
                options,
              });
              return editComponent(
                {
                  data,
                },
                value
              );
            }}
          >
            <Background color="#384E77" variant={BackgroundVariant.Dots} />
            <TorchModelEditorControls
              spacementMultiplierState={[0, () => {}]}
              contentEditable={editable}
              autoLayout={{
                vertical: (n) => applyDagreLayout('TB', n),
                horizontal: (n) => applyDagreLayout('LR', n),
              }}
            />
          </ReactFlow>

          {editable ? (
            <SidebarToggle
              onOpen={() => {
                setIsSidebarOpen(true);
              }}
            />
          ) : null}

          <SidebarBase
            open={isSidebarOpen}
            onClose={() => {
              setIsSidebarOpen(false);
            }}
            tabs={tabs}
            onTabChange={handleSidebarTabChange}
            header={() => {
              switch (currentSidebarTab) {
                case 0:
                  return <Text>You can drag these nodes to the editor</Text>;
                default:
                  return (
                    <Text>You can drag these templates to the editor</Text>
                  );
              }
            }}
          />
        </Box>
        <div>
          {suggestions?.length > 0 && (
            <SuggestionsList suggestions={suggestions} />
          )}
        </div>
      </StyleOverrides>
    </FullScreenWrapper>
  );
};

export default TorchModelEditor;
