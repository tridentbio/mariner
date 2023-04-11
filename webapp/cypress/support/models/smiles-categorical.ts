/// <reference types="cypress" />

import { ModelCreate } from 'app/rtk/generated/models';
import { randomLowerCase } from 'utils';
import { MODEL_SMILES_CATEGORICAL_NAME } from '../constants';
import {
  connectAndFill,
  dragComponents,
  fillDatasetInfo,
  findNotFilled,
  flowDropSelector,
} from './common';

export const buildCategoricalSmilesModel = (
  featureCols: string[],
  targetCol: string
) => {
  cy.visit('/models/new');
  // Dataset information
  fillDatasetInfo({
    modelName: MODEL_SMILES_CATEGORICAL_NAME,
    targetCol: 'tpsa',
    featureCols: ['smiles', 'mwt_group'],
  });

  cy.get('button').contains('NEXT').click();

  cy.get(flowDropSelector).trigger('wheel', {
    deltaY: 1000,
    wheelDeltaX: 0,
    wheelDeltaY: 5000,
    bubbles: true,
  });
  // drag inputs and outputs to know positions:
  featureCols.forEach((col, idx) =>
    cy
      .wait(333)
      .get(`[data-id="${col}"]`)
      .move(flowDropSelector, idx * 100, 6)
  );
  cy.wait(333).get(`[data-id="${targetCol}"]`).move(flowDropSelector, 600, 100);

  const components = [
    'MoleculeFeaturizer',
    'GCNConv',
    'ReLU',
    'GCNConv',
    'ReLU',
    'GCNConv',
    'ReLU',
    'GlobalPooling',
    'OneHot',
    'Concat',
    'Linear',
  ];
  dragComponents(components);
  cy.wait(1000);
  cy.get('[data-testid="RemoveSharpIcon"]').click();
  return cy
    .get('.react-flow__node.react-flow__node-component')
    .then(($nodes) => {
      const componentTypeByDataId: {
        [key: string]: { type: string; filled: Record<string, boolean> };
      } = {};
      const nodes = $nodes.get();

      nodes.forEach((el) => {
        const id = el.getAttribute('data-id');
        const name = el.innerText;
        if (!id) throw new Error('Node without data-id');
        if (!name) throw new Error('Node without innerText');

        componentTypeByDataId[id] = { type: name, filled: {} };
      });

      let sourceId = 'smiles';
      let targetId = findNotFilled('MoleculeFeaturizer', componentTypeByDataId);
      const moleculeFeaturizerId = targetId;
      let sourceHandle = '';
      let targetHandle = '';

      connectAndFill({
        sourceId,
        targetId,
        argValues: {
          sym_bond_list: true,
        },
        componentTypeByDataId,
      });

      sourceId = targetId;
      targetId = findNotFilled('GCNConv', componentTypeByDataId, 'x');
      sourceHandle = 'x';
      targetHandle = 'x';

      connectAndFill({
        sourceId,
        targetId,
        sourceHandle,
        targetHandle,
        argValues: {
          in_channels: '26',
          out_channels: '64',
        },
        componentTypeByDataId,
      });

      // Connect Molecule Featurizer edge_index to all GCNConv edge_index
      Object.entries(componentTypeByDataId)
        .filter(([_key, value]) => value.type.includes('GCNConv'))
        .forEach(([key]) => {
          connectAndFill({
            sourceId,
            targetId: key,
            sourceHandle: 'edge_index',
            targetHandle: 'edge_index',
            componentTypeByDataId,
          });
        });
      // Connect Molecule Featurizer edge_attr to all GCNConv edge_weight
      Object.entries(componentTypeByDataId)
        .filter(([_key, value]) => value.type.includes('GCNConv'))
        .forEach(([key]) => {
          connectAndFill({
            sourceId,
            targetId: key,
            sourceHandle: 'edge_attr',
            targetHandle: 'edge_weight',
            componentTypeByDataId,
          });
        });

      sourceId = targetId;
      targetId = findNotFilled('ReLU', componentTypeByDataId);

      connectAndFill({ sourceId, targetId, componentTypeByDataId });

      sourceId = targetId;
      targetId = findNotFilled('GCNConv', componentTypeByDataId, 'x');
      targetHandle = 'x';

      connectAndFill({
        sourceId,
        targetId,
        targetHandle,
        argValues: {
          out_channels: '64',
        },
        componentTypeByDataId,
      });

      sourceId = targetId;
      targetId = findNotFilled('ReLU', componentTypeByDataId);

      connectAndFill({ sourceId, targetId, componentTypeByDataId });

      sourceId = targetId;
      targetId = findNotFilled('GCNConv', componentTypeByDataId, 'x');
      targetHandle = 'x';

      connectAndFill({
        sourceId,
        targetId,
        targetHandle,
        argValues: {
          out_channels: '64',
        },
        componentTypeByDataId,
      });

      sourceId = targetId;
      targetId = findNotFilled('ReLU', componentTypeByDataId);

      connectAndFill({ sourceId, targetId, componentTypeByDataId });

      sourceId = targetId;
      targetId = findNotFilled('GlobalPooling', componentTypeByDataId, 'x');
      const globalPoolingId = targetId;

      connectAndFill({
        sourceId: moleculeFeaturizerId,
        targetId: globalPoolingId,
        sourceHandle: 'batch',
        targetHandle: 'batch',
        componentTypeByDataId,
      });

      targetHandle = 'x';
      connectAndFill({
        sourceId,
        targetId,
        targetHandle,
        argValues: { aggr: 'sum' },
        componentTypeByDataId,
      });

      sourceId = targetId;
      targetId = findNotFilled('Concat', componentTypeByDataId);
      targetHandle = 'xs';

      connectAndFill({
        sourceId,
        targetId,
        targetHandle,
        argValues: {
          dim: '-1',
        },
        componentTypeByDataId,
      });

      const concatId = targetId;

      sourceId = 'mwt_group';
      targetId = findNotFilled('OneHot', componentTypeByDataId);
      targetHandle = 'x1';
      connectAndFill({
        sourceId,
        targetId,
        targetHandle,
        componentTypeByDataId,
      });

      sourceId = targetId;
      targetId = concatId;
      targetHandle = 'xs';
      connectAndFill({
        sourceId,
        targetId,
        targetHandle,
        componentTypeByDataId,
      });

      sourceId = targetId;
      targetId = findNotFilled('Linear', componentTypeByDataId);
      connectAndFill({
        sourceId,
        targetId,
        argValues: {
          in_features: (66).toString(),
          out_features: '1',
        },
        componentTypeByDataId,
      });

      sourceId = targetId;
      targetId = 'tpsa';

      connectAndFill({ sourceId, targetId, componentTypeByDataId });

      cy.get('[data-testid="ArrowDownwardIcon"]').click();
    });
};
