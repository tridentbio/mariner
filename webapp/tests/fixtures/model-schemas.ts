import { TorchModelSpec } from '@app/rtk/generated/models';
import fs from 'fs';
import { parse } from 'yaml';

const datapath = `${__dirname}/../data`;

const datadirstats = fs.statSync(datapath);
if (!datadirstats.isDirectory()) {
  console.error('Failed to find data directory');
  process.exit(1);
}

const SMALL_REGRESSOR_PATH = `${datapath}/small_regressor_schema.yaml`;
const SMALL_CLASSIFIER_PATH = `${datapath}/small_classifier_schema.yaml`;
const modelSchemaPaths = [SMALL_REGRESSOR_PATH, SMALL_CLASSIFIER_PATH];

/**
 * Gets the path of model schemas in the tests/data folder
 *
 * @returns {string[]}
 */
export const getModelSchemaPaths = (): string[] => {
  return modelSchemaPaths;
};

const loadJsonFromYamlFile = (yamlPath: string) => {
  return parse(fs.readFileSync(yamlPath).toString());
};

export const getRegressorModelSchema = (): TorchModelSpec => {
  return loadJsonFromYamlFile(SMALL_REGRESSOR_PATH);
};
export const getClassfierModelSchema = (): TorchModelSpec => {
  return loadJsonFromYamlFile(SMALL_CLASSIFIER_PATH);
};

const baseSchema: TorchModelSpec = {
  name: 'asd',
  dataset: {
    name: 'das',
    targetColumns: [
      {
        name: 'mwt_group',
        dataType: {
          classes: { yes: 1, no: 0 },
          domainKind: 'categorical' as const,
        },
        outModule: ''
      },
    ],
    featureColumns: [
      {
        name: 'mwt',
        dataType: { domainKind: 'numeric' as const, unit: 'zeta' },
      },
      {
        name: 'smiles',
        dataType: { domainKind: 'smiles' as const },
      },
    ],
  },
  spec: {
    layers: []
  }
};

const testConcatValidatorValid1: TorchModelSpec = {
  ...baseSchema,
  spec: {
    layers: [
      {
        type: 'torch.nn.Linear',
        forwardArgs: {
          input: '$mwt',
        },
        constructorArgs: {
          in_features: 1,
          out_features: 16,
        },
        name: '1',
      },
      {
        type: 'torch.nn.Sigmoid',
        forwardArgs: {
          input: '$1',
        },
        name: '2',
      },
      {
        type: 'torch.nn.Linear',
        forwardArgs: {
          input: '$mwt',
        },
        constructorArgs: {
          in_features: 1,
          out_features: 16,
        },
        name: '3',
      },
      {
        type: 'torch.nn.Sigmoid',
        forwardArgs: {
          input: '$3',
        },
        name: '4',
      },
      {
        name: '5',
        type: 'fleet.model_builder.layers.Concat',
        forwardArgs: {
          xs: ['$2', '$4'],
        },
        constructorArgs: {
          dim: -1,
        },
      },
    ],
  }
};
export const BrokenSchemas = () => {
  const testLinearValidator1: TorchModelSpec = {
    ...baseSchema,
    spec: {
      layers: [
        {
          type: 'torch.nn.Linear',
          name: '1',
          forwardArgs: { input: '$mwt' },
          constructorArgs: {
            in_features: 2,
            out_features: 8,
          },
        },
        { type: 'torch.nn.Sigmoid', name: '2', forwardArgs: { input: '$1' } },
        {
          type: 'torch.nn.Linear',
          name: '3',
          forwardArgs: { input: '$2' },
          constructorArgs: { in_features: 1, out_features: 32 },
        },
      ],
    }
  };

  const testMolFeaturizer1: TorchModelSpec = {
    ...baseSchema,
    dataset: {
      ...baseSchema.dataset,
      featurizers: [
        {
          type: 'fleet.model_builder.featurizers.MoleculeFeaturizer',
          name: 'feat',
          forwardArgs: { mol: '$mwt' },
          constructorArgs: {
            allow_unknown: false,
            per_atom_fragmentation: false,
            sym_bond_list: false,
          },
        },
      ],
    }
  }

  const testGcnConv: TorchModelSpec = {
    ...baseSchema,
    dataset: {
      ...baseSchema.dataset,
      featurizers: [
        {
          type: 'fleet.model_builder.featurizers.MoleculeFeaturizer',
          name: 'feat',
          forwardArgs: { mol: '$smiles' },
          constructorArgs: {
            sym_bond_list: true,
            allow_unknown: false,
            per_atom_fragmentation: false,
          },
        },
      ],
    }, spec: {
      layers: [
        {
          type: 'torch_geometric.nn.GCNConv',
          name: '1',
          forwardArgs: {
            x: '$feat.x',
            edge_index: '$feat.edge_index',
          },
          constructorArgs: {
            in_channels: 10,
            out_channels: 30,
          },
        },
      ],
    }
  };

  const testConcatValidatorInvalid1: TorchModelSpec = {
    ...baseSchema,
    spec: {
      layers: [
        {
          type: 'torch.nn.Linear',
          forwardArgs: {
            input: '$mwt',
          },
          constructorArgs: {
            in_features: 1,
            out_features: 16,
          },
          name: '1',
        },
        {
          type: 'torch.nn.Sigmoid',
          forwardArgs: {
            input: '$1',
          },
          name: '2',
        },
        {
          type: 'torch.nn.Linear',
          forwardArgs: {
            input: '$mwt',
          },
          constructorArgs: {
            in_features: 1,
            out_features: 17,
          },
          name: '3',
        },
        {
          type: 'torch.nn.Sigmoid',
          forwardArgs: {
            input: '$3',
          },
          name: '4',
        },
        {
          name: '5',
          type: 'fleet.model_builder.layers.Concat',
          forwardArgs: {
            xs: ['$2', '$4'],
          },
          constructorArgs: {
            dim: 0,
          },
        },
      ],
    }
  };
  return {
    testMolFeaturizer1,
    testLinearValidator1,
    testGcnConv,
    testConcatValidatorInvalid1,
  } as const;
};

/**
 * Gets valid model schemas examples
 *
 * @returns {TorchModelSpec} [TODO:description]
 */
export const getValidModelSchemas = (): TorchModelSpec[] => {
  return modelSchemaPaths
    .map(loadJsonFromYamlFile)
    .concat(testConcatValidatorValid1);
};
