import { ThemeProvider } from '@emotion/react';
import { Meta } from '@storybook/react';
import { store } from 'app/store';
import { ModelEditorContextProvider } from 'hooks/useModelEditor';
import { ModelSchema } from 'model-compiler/src/interfaces/model-editor';
import { useState } from 'react';
import { ReactFlowProvider } from 'react-flow-renderer';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { theme } from 'theme';
import ModelEditor from '.';

const storyMeta: Meta = {
  title: 'Components/Model Editor v2',
  component: ModelEditor,
  decorators: [
    (Story) => {
      return (
        <ThemeProvider theme={theme}>
          <Provider store={store}>
            <BrowserRouter>
              <div style={{ width: '100vw', height: '100vh' }}>
                <ReactFlowProvider>
                  <ModelEditorContextProvider>
                    {Story()}
                  </ModelEditorContextProvider>
                </ReactFlowProvider>
              </div>
            </BrowserRouter>
          </Provider>
        </ThemeProvider>
      );
    },
  ],
};
export default storyMeta;
export const Default = () => {
  const [modelSchema, setModelSchema] = useState<ModelSchema>({
    name: 'asdasd',
    dataset: {
      name: 'akakak',
      featureColumns: [
        { name: 'mwt', dataType: { domainKind: 'numeric', unit: 'mole' } },
        { name: 'smiles', dataType: { domainKind: 'smiles' } },
      ],
      targetColumns: [
        {
          name: 'exp',
          dataType: { domainKind: 'numeric', unit: 'mole' },
        },
      ],
    },
    layers: [],
    featurizers: [],
  });
  return <ModelEditor value={modelSchema} onChange={setModelSchema} />;
};

export const InvalidSchema = () => {
  const [modelSchema, setModelSchema] = useState<ModelSchema>({
    name: 'asdasd',
    dataset: {
      name: 'akakak',
      featureColumns: [
        { name: 'mwt', dataType: { domainKind: 'numeric', unit: 'mole' } },
        { name: 'smiles', dataType: { domainKind: 'smiles' } },
      ],
      targetColumns: [
        {
          name: 'exp',
          dataType: { domainKind: 'numeric', unit: 'mole' },
        },
      ],
    },
    layers: [
      {
        name: 'Linear1',
        type: 'torch.nn.Linear',
        constructorArgs: { in_features: 0, out_features: 0 },
        forwardArgs: { input: '$mwt' },
      },
      {
        name: 'GCNConv2',
        type: 'torch_geometric.nn.GCNConv',
        constructorArgs: { in_channels: 0, out_channels: 0 },
        forwardArgs: {
          x: '$MolFeaturizer2.x',
          edge_index: '',
          edge_weight: '',
        },
      },
    ],
    featurizers: [
      {
        name: 'MolFeaturizer1',
        type: 'model_builder.featurizers.MoleculeFeaturizer',
        constructorArgs: {
          sym_bond_list: false,
          allow_unknown: false,
          per_atom_fragmentation: false,
        },
        forwardArgs: { mol: `$mwt` },
      },
      {
        name: 'MolFeaturizer2',
        type: 'model_builder.featurizers.MoleculeFeaturizer',
        constructorArgs: {
          sym_bond_list: false,
          allow_unknown: false,
          per_atom_fragmentation: false,
        },
        forwardArgs: { mol: `$smiles` },
      },
    ],
  });
  return <ModelEditor value={modelSchema} onChange={setModelSchema} />;
};
