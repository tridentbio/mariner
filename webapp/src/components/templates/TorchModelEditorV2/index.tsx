import { useTorchModelEditor } from '@hooks';
import { Box } from '@mui/material';
import FullScreenWrapper from 'components/organisms/FullScreenWrapper';
import TorchModelEditorControls from 'components/templates/TorchModelEditor/Components/TorchModelEditorControls';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import {
  getComponent,
  getNode,
} from 'model-compiler/src/implementation/modelSchemaQuery';
import {
  ModelSchema,
  NodeType,
} from '@model-compiler/src/interfaces/torch-model-editor';
import {
  DragEvent,
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import ReactFlow, { Background, BackgroundVariant, NodeProps } from 'reactflow';
import 'reactflow/dist/style.css';
import { substrAfterLast } from 'utils';
import OptionsSidebarV2, {
  HandleProtoDragStartParams,
} from './OptionsSidebarV2';
import SuggestionsList from './SuggestionsList';
import ModelEdge from './edges/ModelEdge';
import ComponentConfigNode from './nodes/ComponentConfigNode';
import InputNode from './nodes/InputNode';
import OutputNode from './nodes/OutputNode';
import { StyleOverrides } from './styles';

type TorchModelEditorProps = {
  value: ModelSchema;
  onChange?: (schema: ModelSchema) => void;
  editable?: boolean;
  dagre?: number | 'goodDistance';
};

const getGoodDistance = (nodesNumber: number) => {
  if (nodesNumber < 3) return 8;
  else if (nodesNumber < 5) return 6;
  else if (nodesNumber < 7) return 3;
  else return 1;
};

const SideBar = memo(
  ({ editable }: { editable: boolean }) => (
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
            x: event.clientX - event.currentTarget.getBoundingClientRect().x,
            y: event.clientY - event.currentTarget.getBoundingClientRect().y,
          })
        );
        event.dataTransfer.effectAllowed = 'move';
      }}
      editable={editable}
    />
  ),
  (prevValue, nextValue) => prevValue.editable === nextValue.editable
);

SideBar.displayName = 'SideBar';

const TorchModelEditor = ({
  value,
  onChange,
  editable = true,
  dagre,
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

  useEffect(() => {
    if (value) {
      setSchema(value);
    }
  }, []);

  useEffect(() => {
    fitView();
  }, [nodesInitialized]);

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

    if (
      componentData.type === 'scikit_reg' ||
      componentData.type === 'scikit_class'
    ) {
      return;
    }
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
            position: 'relative',
          }}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            minZoom={0.1}
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
              return editComponent({
                schema: value,
                data,
              });
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

          {editable ? <SideBar editable /> : null}
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
