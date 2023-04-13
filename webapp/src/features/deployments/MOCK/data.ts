import { DataTypeDomainKind } from '@app/types/domain/datasets';
import { EClassPaths } from '@app/types/domain/modelOptions';
import {
  EDeploymnetStatuses,
  ERateLimitUnits,
  EShareStrategies,
} from '../types';
const modelVersionData = {
  id: 1,
  modelId: 1,
  name: 'debt-1',
  description: 'Description',
  config: {
    name: 'GNNExample',
    dataset: {
      name: 'Small Zinc dataset',
      targetColumns: [
        {
          name: 'tpsa',
          dataType: {
            domainKind: DataTypeDomainKind.Numerical,
            unit: 'mole',
          },
        },
      ],
      featureColumns: [
        {
          name: 'smiles',
          dataType: {
            domainKind: DataTypeDomainKind.Smiles,
          },
        },
        {
          name: 'mwt',
          dataType: {
            domainKind: DataTypeDomainKind.Numerical,
          },
        },
      ],
    },
    layers: [
      {
        type: EClassPaths.GCN_CONV,
        name: 'GCN1',
        constructorArgs: {
          in_channels: 26,
          out_channels: 64,
          improved: false,
          cached: false,
          add_self_loops: false,
          normalize: false,
          bias: false,
        },
        forwardArgs: {
          x: '${MolToGraphFeaturizer.x}',
          edge_index: '${MolToGraphFeaturizer.edge_index}',
          edge_weight: '${MolToGraphFeaturizer.edge_attr}',
        },
      },
      {
        type: EClassPaths.RELU,
        name: 'GCN1_Activation',
        constructorArgs: {
          inplace: false,
        },
        forwardArgs: {
          input: '${GCN1}',
        },
      },
      {
        type: EClassPaths.GCN_CONV,
        name: 'GCN2',
        constructorArgs: {
          in_channels: 64,
          out_channels: 64,
          improved: false,
          cached: false,
          add_self_loops: false,
          normalize: false,
          bias: false,
        },
        forwardArgs: {
          x: '${GCN1_Activation}',
          edge_index: '${MolToGraphFeaturizer.edge_index}',
          edge_weight: '${MolToGraphFeaturizer.edge_attr}',
        },
      },
      {
        type: EClassPaths.RELU,
        name: 'GCN2_Activation',
        constructorArgs: {
          inplace: false,
        },
        forwardArgs: {
          input: '${GCN2}',
        },
      },
      {
        type: EClassPaths.GCN_CONV,
        name: 'GCN3',
        constructorArgs: {
          in_channels: 64,
          out_channels: 64,
          improved: false,
          cached: false,
          add_self_loops: false,
          normalize: false,
          bias: false,
        },
        forwardArgs: {
          x: '${GCN2_Activation}',
          edge_index: '${MolToGraphFeaturizer.edge_index}',
          edge_weight: '${MolToGraphFeaturizer.edge_attr}',
        },
      },
      {
        type: EClassPaths.RELU,
        name: 'GCN3_Activation',
        constructorArgs: {
          inplace: false,
        },
        forwardArgs: {
          input: '${GCN3}',
        },
      },
      {
        type: EClassPaths.GLOBAL_POOLING,
        name: 'AddPool',
        constructorArgs: {
          aggr: 'sum',
        },
        forwardArgs: {
          x: '${GCN3_Activation}',
          batch: '${MolToGraphFeaturizer.batch}',
          size: undefined,
        },
      },
      {
        type: EClassPaths.LINEAR,
        name: 'Linear1',
        constructorArgs: {
          in_features: 1,
          out_features: 10,
          bias: false,
        },
        forwardArgs: {
          input: '${mwt}',
        },
      },
      {
        type: EClassPaths.CONCAT,
        name: 'Combiner',
        constructorArgs: {
          dim: -1,
        },
        forwardArgs: {
          xs: ['${AddPool}', '${Linear1}'],
        },
      },
      {
        type: EClassPaths.LINEAR,
        name: 'LinearJoined',
        constructorArgs: {
          in_features: 74,
          out_features: 1,
          bias: false,
        },
        forwardArgs: {
          input: '${Combiner}',
        },
      },
    ],
    featurizers: [
      {
        type: EClassPaths.MOLECULE_FEATURIZER,
        name: 'MolToGraphFeaturizer',
        constructorArgs: {
          allow_unknown: false,
          sym_bond_list: true,
          per_atom_fragmentation: false,
        },
        forwardArgs: {
          mol: '${smiles}',
        },
      },
    ],
  },
  createdAt: new Date('2022-10-21T14:51:10.259278+00:00').toString(),
  updatedAt: new Date('2022-10-21T14:51:10.259278').toString(),
};

export const getDeploymentsData = [
  {
    id: 1,
    name: 'Deployment Name',
    readme: '',
    shareUrl: '',
    status: EDeploymnetStatuses.ACTIVE,
    modelVersionId: 2,
    modelVersion: modelVersionData,
    shareStrategy: EShareStrategies.PUBLIC,
    organizationsAllowed: [],
    usersIdAllowed: [],
    showTrainingData: true,
    predictionRateLimitValue: 100,
    predictionRateLimitUnit: ERateLimitUnits.DAY,
    createdByUserId: 2,
    createdAt: new Date('2022-11-21T14:51:10.259278+00:00'),
    isDeleted: false,
  },
  {
    id: 2,
    name: 'Deployment Name2',
    readme: '',
    shareUrl: '',
    status: EDeploymnetStatuses.IDLE,
    modelVersionId: 2,
    modelVersion: modelVersionData,
    shareStrategy: EShareStrategies.PRIVATE,
    organizationsAllowed: ['@mariner.com'],
    usersIdAllowed: [],
    showTrainingData: true,
    predictionRateLimitValue: 100,
    predictionRateLimitUnit: ERateLimitUnits.MINUTE,
    createdByUserId: 2,
    createdAt: new Date('2022-11-21T14:51:10.259278+00:00'),
    isDeleted: false,
  },
  {
    id: 3,
    name: 'Deployment Name3',
    readme: '',
    shareUrl: '',
    status: EDeploymnetStatuses.STOPPED,
    modelVersionId: 2,
    modelVersion: modelVersionData,
    shareStrategy: EShareStrategies.PRIVATE,
    organizationsAllowed: [],
    usersIdAllowed: [4, 7],
    usersAllowed: [
      {
        email: 'user1@email.com',
        full_name: 'User1',
        id: 4,
      },
      {
        email: 'user2@email.com',
        full_name: 'User2',
        id: 7,
      },
    ],
    showTrainingData: true,
    predictionRateLimitValue: 100,
    predictionRateLimitUnit: ERateLimitUnits.HOUR,
    createdByUserId: 2,
    createdAt: new Date('2022-11-21T14:51:10.259278+00:00'),
    isDeleted: false,
  },
];
