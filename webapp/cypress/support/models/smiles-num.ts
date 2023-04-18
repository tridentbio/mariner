/// <reference types="cypress" />

import { parse } from 'yaml';
import { DeepPartial } from '@reduxjs/toolkit';
import { ModelCreate, ModelSchema } from '@app/rtk/generated/models';
import { randomLowerCase } from 'utils';
import { dragComponentsAndMapConfig, flowDropSelector } from './common';
import { iterateTopologically, unwrapDollar } from 'model-compiler/src/utils';
import { NodeType } from 'model-compiler/src/interfaces/model-editor';

const randomName = () => randomLowerCase(8);

const getTypeByName = (config: ModelSchema, name: string): string => {
  const [layer] = config.layers!.filter((layer) => layer.name === name);
  if (layer) return layer.type;
  const [featurizer] = config.featurizers!.filter(
    (featurizer) => featurizer.name === name
  );
  if (featurizer) return featurizer.type;

  return name;
};

// Returns [sourceHandleId, targetHandleId][]
const getIncomingEdges = (
  node: NodeType,
  config: ModelSchema
): [string, string][] => {
  if (node.type === 'input' /*  || node.type === 'output' */) return [];
  if (node.type === 'output')
    return [[node.name, getTypeByName(config, node.outModule!)]];

  return Object.entries(node.forwardArgs || {}).reduce(
    (acc, [forwardArg, targetHandleIds]) => {
      if (!targetHandleIds) return acc;
      if (!Array.isArray(targetHandleIds)) targetHandleIds = [targetHandleIds];

      return [
        ...acc,
        ...targetHandleIds.map((targetHandleId: string) => {
          const [targetOriginalName, ...tail] =
            unwrapDollar(targetHandleId).split('.');

          return [
            node.type + '.' + forwardArg,
            getTypeByName(config, targetOriginalName) +
              (tail.length ? '.' + tail.join('.') : ''),
          ];
        }),
      ];
    },
    [] as [string, string][]
  );
};

const parseEdgeName = (edgeName: string) => {
  let [sourceNodeId, ...tailSource] = edgeName.split('.').reduce((acc, cur) => {
    if (acc.length) acc.push(cur);
    else if (cur.match(/-\d+$/g)) {
      acc.push(cur);
    }
    return acc;
  }, [] as string[]);

  if (sourceNodeId) return [sourceNodeId, tailSource.join('.')];
  return [edgeName, ''];
};

const objIsEmpty = (obj: object) => !Boolean(Object.keys(obj).length);

const autoFixSuggestions = () =>
  cy.getWithouThrow('[data-testid="AutoFixHighOutlinedIcon"]').then(($els) => {
    for (let i = 0; i < $els.length; i++) {
      cy.get('[data-testid="AutoFixHighOutlinedIcon"]')
        .first()
        .click({ force: true });
      cy.wait(100);
    }
  });

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

  const [sourceNodeId, tailSource] = parseEdgeName(sourceHandleId);
  // sourceHandleId = tailSource.join('.');
  const [targetNodeId, tailTarget] = parseEdgeName(targetHandleId);
  // targetHandleId = tailTarget.join('.');
  const makeNodeSelector = (nodeId: string, handleId?: string) => {
    const selector =
      nodeId && handleId
        ? `[data-nodeid="${nodeId}"][data-handleid="${handleId}"]`
        : `[data-nodeid="${nodeId}"]`;

    const element = cy.get(selector);
    if (!handleId)
      return element.filter((_, el) => !el.hasAttribute('data-handleid'));

    return element;
  };
  const target = makeNodeSelector(targetNodeId, tailTarget);
  const source = makeNodeSelector(sourceNodeId, tailSource);

  target.trigger('mousedown', {
    button: 0,
  });
  source.trigger('mousemove');
  source.trigger('mouseup');
};

export const buildModel = (modelCreate: DeepPartial<ModelCreate>) => {
  cy.once('uncaught:exception', () => false);
  cy.visit('/models/new');
  // Fill model name
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

  const targetCols =
    modelCreate.config?.dataset?.targetColumns?.map((col) => col?.name || '') ||
    [];
  const featureCols =
    modelCreate.config?.dataset?.featureColumns?.map(
      (col) => col?.name || ''
    ) || [];

  targetCols.forEach((col) => {
    cy.get('#target-col').click();
    cy.get('div').contains(col).click({ force: true });
  });
  featureCols.forEach((col) => {
    cy.get('#feature-cols').click();
    cy.get('div').contains(col).click({ force: true });
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

  const config = dragComponentsAndMapConfig(
    modelCreate.config as unknown as ModelSchema
  );
  cy.get('div[aria-label="Apply auto vertical layout"] button').click();
  cy.get('button[title="fit view"]').click();
  cy.get('button[aria-label="Close all components"]').click();
  cy.get('button[aria-label="Open all components"]').click();

  cy.then(() =>
    iterateTopologically(config, (node, type) => {
      const edges: [string, string][] = getIncomingEdges(node, config);
      edges.forEach(([sourceHandleId, targetHandleId]) =>
        connect(sourceHandleId, targetHandleId)
      );
    })
  );

  cy.then(() => {
    iterateTopologically(config, (node, type) => {
      if (['input', 'output'].includes(type)) return;
      const args = 'constructorArgs' in node ? node.constructorArgs : {};
      if (objIsEmpty(args)) return;

      Object.entries(args).forEach(([key, value]) => {
        autoFixSuggestions().then(() => {
          const element = cy
            .get(`[data-id="${parseEdgeName(node.type)[0]}"]`)
            .first();

          const isBool = !['number', 'string'].includes(typeof value);

          if (!isBool) {
            const curElement = element
              .contains('label', key)
              .next('div')
              .find('input');
            curElement.clear().type(value);
          } else {
            element.get(`[id="${key}"]`).then((curElement) => {
              if (Boolean(curElement.prop('checked')) !== value)
                curElement.click();
            });
          }
        });
      });
    });
  }).then(() => cy.get('button').contains('CREATE').click());

  cy.get('.react-flow__node').then(($nodes) => {
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

export const buildNumSmilesModel = (dataset: string | null = null) => {
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
          name: dataset || 'ZincExtra',
          featureColumns: [{ name: 'smiles' }, { name: 'mwt' }],
          targetColumns: [{ name: 'tpsa', outModule: 'LinearJoined' }],
        },
      },
    });
  });
};

export const buildYamlModel = (yaml: string, dataset: string | null = null) => {
  cy.fixture(yaml).then((yamlStr) => {
    const jsonSchema: ModelSchema = parse(yamlStr);
    buildModel({
      name: randomName(),
      modelVersionDescription: randomName(),
      modelDescription: randomName(),
      config: {
        ...jsonSchema,
        name: randomName(),
        dataset: {
          ...jsonSchema.dataset,
          name: dataset || jsonSchema.dataset.name,
        },
      },
    });
  });
};
