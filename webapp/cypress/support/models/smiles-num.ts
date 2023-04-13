/// <reference types="cypress" />

import { parse } from 'yaml';
import { DeepPartial } from '@reduxjs/toolkit';
import { ModelCreate, ModelSchema } from '@app/rtk/generated/models';
import { randomLowerCase } from 'utils';
import { dragComponentsAndMapConfig, flowDropSelector } from './common';
import { iterateTopologically, unwrapDollar } from 'model-compiler/src/utils';
import { NodeType } from 'model-compiler/src/interfaces/model-editor';

const randomName = () => randomLowerCase(8);

// Returns [sourceHandleId, targetHandleId][]
const getIncomingEdges = (node: NodeType): [string, string][] => {
  if (node.type === 'input' || node.type === 'output') return [];
  return Object.entries(node.forwardArgs || {}).reduce(
    (acc, [forwardArg, targetHandleId]) => {
      if (!targetHandleId) return acc;
      return [
        ...acc,
        [node.name + '.' + forwardArg, unwrapDollar(targetHandleId)],
      ];
    },
    [] as [string, string][]
  );
};

/**
 * Uses cy to connect a node handle to a target handle.
 *
 * handleIds are <node.id>.<arg_name> if the arg is not from a simple output
 * in which case it is simply <node.id>.
 *
 *
 * @param {string} sourceHandleId - source handleId as described
 * @param {string} targetHandleId - target handleId
 */
const connect = (sourceHandleId: string, targetHandleId: string) => {
  Cypress.log({
    name: 'connect',
    displayName: `connect("${sourceHandleId}", "${targetHandleId}")`,
  });
  const [sourceNodeId, ...tailSource] = sourceHandleId.split('.');
  sourceHandleId = tailSource.join('.');
  const [targetNodeId, ...tailTarget] = targetHandleId.split('.');
  targetHandleId = tailTarget.join('.');
  const makeNodeSelector = (nodeId: string, handleId?: string) =>
    nodeId && handleId
      ? `[data-nodeid="${nodeId}"][data-handleid="${handleId}"]`
      : `data-nodeid="${nodeId}"`;
  cy.get(makeNodeSelector(targetNodeId, targetHandleId)).move(
    makeNodeSelector(sourceNodeId, sourceHandleId),
    1,
    1
  );
};

export const buildModel = (modelCreate: DeepPartial<ModelCreate>) => {
  cy.visit('/models/new');

  // Fill model name
  // TODO: use typing and select no click
  //cy.get('[data-testid="model-name"]')
  //  .click()
  //  .type(modelCreate.name || randomName())
  //  .get('li[role="option"]')
  //  .first()
  //  .click();
  cy.get('[data-testid="random-model-name"]').click();
  // Fill model description
  cy.get('[data-testid="model-description"] input')
    .clear()
    .type(modelCreate.modelDescription || randomName());
  // Fill model version name
  cy.get('[data-testid="version-name"] input')
    .clear()
    .type(modelCreate?.config?.name || randomName());
  // Fill model version description
  cy.get('[data-testid="version-description"] textarea')
    .clear()
    .type(modelCreate.modelVersionDescription || randomName());

  cy.get('button').contains('NEXT').click();

  cy.get('#dataset-select')
    .click()
    .type(modelCreate.config?.dataset?.name || '')
    .get('li[role="option"]')
    .first()
    .click();
  const targetCol =
    (modelCreate.config?.dataset?.targetColumns?.length &&
      modelCreate.config?.dataset?.targetColumns[0] &&
      modelCreate.config?.dataset?.targetColumns[0].name) ||
    '';
  const featureCols =
    modelCreate.config?.dataset?.featureColumns?.map(
      (col) => col?.name || ''
    ) || [];
  cy.get('#target-col').click();
  cy.get('div').contains(targetCol).click();
  featureCols.forEach((col) => {
    cy.get('#feature-cols').click();
    cy.get('div').contains(col).click();
  });

  cy.get('button').contains('NEXT').click();

  cy.get(flowDropSelector).trigger('wheel', {
    deltaY: 0,
    wheelDeltaX: 0,
    wheelDeltaY: 2500,
    bubbles: true,
  });
  // drag inputs and outputs to know positions:
  featureCols.forEach((col, idx) =>
    cy
      .wait(333)
      .get(`[data-id="${col}"]`)
      .move(flowDropSelector, idx * 100, 6)
  );
  cy.wait(334).get(`[data-id="${targetCol}"]`).move(flowDropSelector, 500, 150);

  const config = dragComponentsAndMapConfig(
    modelCreate.config as unknown as ModelSchema
  );

  iterateTopologically(config, (node, type) => {
    const edges: [string, string][] = getIncomingEdges(node);
    edges.forEach(([sourceHandleId, targetHandleId]) =>
      connect(sourceHandleId, targetHandleId)
    );
  });

  return cy.get('.react-flow__node').then(($nodes) => {
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
  });
};

export const buildNumSmilesModel = () => {
  cy.fixture('models/schemas/small_regressor_schema.yaml').then((yamlStr) => {
    const jsonSchema = parse(yamlStr);
    buildModel({
      name: 'asidjaisjd',
      modelVersionDescription: 'AAAAAAAAAAAAAAAAAAAAa',
      modelDescription: 'BBBBBBBBBBBBBBBB',
      config: {
        ...jsonSchema,
        name: 'CCCCCCCCCCCCCC',
        dataset: {
          name: 'ZincExtra',
          featureColumns: [{ name: 'smiles' }, { name: 'mwt' }],
          targetColumns: [{ name: 'tpsa' }],
        },
      },
    });
  });
};
