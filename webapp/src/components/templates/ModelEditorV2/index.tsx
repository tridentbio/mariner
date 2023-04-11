import { useEffect, useRef, DragEvent, useState, useLayoutEffect } from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  NodeProps,
  applyNodeChanges,
  applyEdgeChanges,
} from 'react-flow-renderer';
import useModelEditor from 'hooks/useModelEditor';
import ModelEditorControls from 'components/templates/ModelEditor/Components/ModelEditorControls';
import { Box } from '@mui/material';
import OptionsSidebarV2, {
  HandleProtoDragStartParams,
} from './OptionsSidebarV2';
import { substrAfterLast } from 'utils';
import ComponentConfigNode from './nodes/ComponentConfigNode';
import InputNode from './nodes/InputNode';
import OutputNode from './nodes/OutputNode';
import SuggestionsList from './SuggestionsList';
import ModelEdge from './edges/ModelEdge';
import { StyleOverrides } from './styles';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import {
  getComponent,
  getNode,
} from 'model-compiler/src/implementation/modelSchemaQuery';
import {
  ModelSchema,
  NodeType,
} from 'model-compiler/src/interfaces/model-editor';
import { useMemo } from 'react';
import FullScreenWrapper from 'components/organisms/FullScreenWrapper';

type ModelEditorProps = {
  value: ModelSchema;
  onChange?: (schema: ModelSchema) => void;
  editable?: boolean;
};

const ModelEditor = ({
  value,
  onChange,
  editable = true,
}: ModelEditorProps) => {
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
  } = useModelEditor();
  const [fullScreen, setFullScreen] = useState(false);
  const [connectingNode, setConnectingNode] = useState<
    | {
        component: NodeType;
        handleId: string | null;
        handleType: 'source' | 'target';
      }
    | undefined
  >();

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

  useEffect(() => {
    if (value) {
      setSchema(value);
    }
  }, []);

  const reactFlowWrapper = useRef<HTMLDivElement | null>(null);
  const [firstRenderWithElements, setFirstRenderWithElements] = useState(false);
  useEffect(() => {
    if (!reactFlowWrapper.current) return;
    const [nodes, edges] = schemaToEditorGraph(value);
    if (editable) {
      setNodes(nodes.reverse());
      setEdges(edges);
      getSuggestions({ schema: value });
      applyDagreLayout('LR', 5);
    } else {
      setNodes(nodes);
      const outputEdges = [];
      for (const targetColumn of value.dataset.targetColumns) {
        outputEdges.push({
          type: 'ModelEdge',
          id: `output-edge-${targetColumn.name}`,
          source: targetColumn.outModule || nodes.at(-1)?.id || '',
          target: targetColumn.name,
        });
      }
      setEdges([...edges, ...outputEdges]);
      getSuggestions({ schema: value });
      applyDagreLayout('TB', 5);
    }
  }, []);

  // TODO find a better way to fix screen initial state
  const temporalyFixScreen = async () => {
    // temporary fix for screen not rendering correctly
    const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));
    await sleep(500);
    applyDagreLayout('LR', 3);
    await sleep(100);
    setFirstRenderWithElements(true);
  };
  useEffect(() => {
    if (!editable) temporalyFixScreen();
  }, []);
  useEffect(() => {
    if (!editable) fitView();
  }, [firstRenderWithElements]);

  useEffect(() => {
    onChange && schema && onChange(schema);
  }, [schema]);

  /**
   * DragOver event handler for when an option from the options sidebar
   * is dragged over the canvas.
   */
  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  };

  const [componentsDropped, setComponentsDropped] = useState(0);
  /**
   * Drop event handler, with the side effect of adding the node with empty
   * arguments in the canvas state
   */
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    if (!reactFlowWrapper.current) return;
    const componentData = JSON.parse(
      event.dataTransfer.getData('application/componentData')
    ) as HandleProtoDragStartParams['data'];
    const offset = JSON.parse(event.dataTransfer.getData('application/offset'));
    const { type: classPath } = componentData.component;
    const { type: componentType } = componentData;
    if (!classPath) return;
    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
    const position = project({
      x: event.clientX - reactFlowBounds.left - offset.x,
      y: event.clientY - reactFlowBounds.top - offset.y,
    });
    addComponent(
      {
        schema: value,
        name: `${substrAfterLast(classPath, '.')}-${componentsDropped}`,
        type: classPath,
        componentType,
      },
      position
    );
    setComponentsDropped((dropped) => dropped + 1);
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
    editComponent({
      schema,
      data,
    });
  };
  useLayoutEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [listeningForKeyPress, keyAssignments]);
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
          }}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            minZoom={0.1}
            onDragOver={onDragOver}
            onDrop={onDrop}
            onNodesChange={(changes) => {
              setNodes((nodes) => applyNodeChanges(changes, nodes));
            }}
            onEdgesChange={(changes) =>
              setEdges((edges) => applyEdgeChanges(changes, edges))
            }
            onConnectEnd={() => {
              clearPositionOrdering();
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
              return editComponent({
                schema: value,
                data,
              });
            }}
          >
            <Background color="#384E77" variant={BackgroundVariant.Dots} />
            <ModelEditorControls
              spacementMultiplierState={[0, () => {}]}
              contentEditable={editable}
              autoLayout={{
                vertical: (n) => applyDagreLayout('TB', n),
                horizontal: (n) => applyDagreLayout('LR', n),
              }}
            />
          </ReactFlow>

          {editable && (
            <OptionsSidebarV2
              onDragStart={({ event, data }: HandleProtoDragStartParams) => {
                event.dataTransfer.setData(
                  'application/reactflow',
                  'ComponentConfigNode'
                );
                event.dataTransfer.setData(
                  'application/componentData',
                  JSON.stringify(data)
                );
                event.dataTransfer.setData(
                  'application/offset',
                  JSON.stringify({
                    x:
                      event.clientX -
                      event.currentTarget.getBoundingClientRect().x,
                    y:
                      event.clientY -
                      event.currentTarget.getBoundingClientRect().y,
                  })
                );
                event.dataTransfer.effectAllowed = 'move';
              }}
              editable={editable}
            />
          )}
        </Box>
        <div>
          {suggestions.length > 0 && (
            <SuggestionsList suggestions={suggestions} />
          )}
        </div>
      </StyleOverrides>
    </FullScreenWrapper>
  );
};

export default ModelEditor;
