/// <reference types="cypress" />

import { parse } from 'yaml';
import { DeepPartial } from '@reduxjs/toolkit';
import { ModelCreate, TorchModelSpec } from '@app/rtk/generated/models';
import { randomLowerCase } from 'utils';
import { dragComponentsAndMapConfig, flowDropSelector } from './common';
import {
  extendSpecWithTargetForwardArgs,
  iterateTopologically,
  unwrapDollar,
} from 'model-compiler/src/utils';
import { NodeType } from 'model-compiler/src/interfaces/model-editor';
import { SOME_MODEL_NAME } from '../constants';

const randomName = () => randomLowerCase(8);

const getTypeByName = (config: TorchModelSpec, name: string): string => {
  const [layer] = config.spec.layers!.filter((layer) => layer.name === name);
  if (layer) return layer.type;
  const [featurizer] = config.dataset.featurizers!.filter(
    (featurizer) => featurizer.name === name
  );
  if (featurizer?.type) return featurizer.type;
  return name;
};

// Returns [sourceHandleId, targetHandleId][]
const getIncomingEdges = (
  node: NodeType,
  config: TorchModelSpec
): [string, string][] => {
  if (node.type === 'input') return [];
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
  cy.getWithoutThrow('[data-testid="AutoFixHighOutlinedIcon"]').then(($els) => {
    if ($els.length === 1)
      cy.get('[data-testid="AutoFixHighOutlinedIcon"]')
        .first()
        .click({ force: true });
    else if ($els.length !== 0) cy.get('li').contains('Fix all').click();
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

export const buildModel = (
  modelCreate: DeepPartial<ModelCreate>,
  success: boolean
) => {
  cy.once('uncaught:exception', () => false);
  cy.visit('/models/new');
  // Fill model name
  cy.get('[data-testid="model-name"] input')
    .clear()
    .type(modelCreate.name || randomName())
    .type('{enter}');

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
    modelCreate.config?.dataset?.targetColumns?.map(
      // domainKind used to determine end of column name
      (col) => col!.name! + col!.dataType!.domainKind!
    ) || [];

  const featureCols =
    modelCreate.config?.dataset?.featureColumns?.map(
      // domainKind used to determine end of column name
      (col) => col!.name! + col!.dataType!.domainKind!
    ) || [];

  targetCols.forEach((col) => {
    cy.get('#target-col').click();
    cy.get('li')
      .contains(new RegExp(`^${col}`, 'gi'))
      .click();
  });

  featureCols.forEach((col) => {
    cy.get('#feature-cols').click();
    cy.get('li')
      .contains(new RegExp(`^${col}`, 'gi'))
      .click();
  });

  cy.get('button').contains('NEXT').click();

  cy.get(flowDropSelector).trigger('wheel', {
    deltaY: 0,
    wheelDeltaX: 0,
    wheelDeltaY: 2500,
    bubbles: true,
  });

  const mod = extendSpecWithTargetForwardArgs(
    modelCreate.config as unknown as TorchModelSpec
  );

  cy.log('Dragging components');
  cy.log(
    'Total layers + featurizers',
    (mod.spec?.layers?.length || 0) + (mod.dataset?.targetColumns?.length || 0)
  );

  const config = dragComponentsAndMapConfig(mod);

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

  cy.intercept({
    method: 'POST',
    url: 'http://localhost/api/v1/models/check-config',
  }).as('checkConfig');
  cy.intercept({
    method: 'POST',
    url: 'http://localhost/api/v1/models',
  }).as('createModel');

  cy.then(() => {
    iterateTopologically(config, (node, type) => {
      if (['input', 'output'].includes(type)) return;
      const args = 'constructorArgs' in node ? node.constructorArgs : {};
      if (objIsEmpty(args)) return;

      Object.entries(args).forEach(([key, value]) => {
        autoFixSuggestions().then(() => {
          const element = cy
            .get(`[data-id="${parseEdgeName(node.type!)[0]}"]`)
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
                curElement.trigger('click');
            });
          }
        });
      });
    });
  }).then(() => cy.get('button').contains('CREATE').click());

  cy.wait('@checkConfig', { timeout: 15 }).then(({ response }) => {
    expect(response?.statusCode).to.eq(200);
    if (success) {
      expect(Boolean(response?.body.stackTrace)).to.eq(false);
    }
  });

  if (success)
    cy.wait('@createModel').then(({ response }) => {
      expect(response?.statusCode).to.eq(200);
    });
};

export const modelExists = (name: string): Cypress.Chainable<boolean> =>
  cy.getCurrentAuthString().then((authorization) =>
    cy
      .request({
        method: 'GET',
        url: `http://localhost/api/v1/models?page=0&perPage=50&q=${name}`,
        headers: {
          authorization,
        },
      })
      .then((response) => {
        expect(response?.status).to.eq(200);
        const models: any[] =
          'data' in response?.body ? response.body.data : [];
        return cy.wrap(models.some((model) => model.name === name));
      })
  );

export const buildYamlModel = (
  yaml: string,
  dataset: string | null = null,
  success = true,
  deleteModel = true,
  modelName = randomName()
) =>
  cy.fixture(yaml).then((yamlStr) => {
    const jsonSchema: TorchModelSpec = parse(yamlStr);

    return buildModel(
      {
        name: modelName,
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
      },
      success
    );
  });

export const setupSomeModel = () =>
  modelExists(SOME_MODEL_NAME).then((exists) => {
    cy.on('uncaught:exception', () => false);
    if (exists) {
      cy.log('Model already exists, skipping creation');
      return cy.wrap(SOME_MODEL_NAME);
    }

    return cy.setupIrisDatset().then((irisDataset) => {
      buildYamlModel(
        'data/yaml/binary_classification_model.yaml',
        irisDataset.name,
        true,
        true,
        SOME_MODEL_NAME
      );
      return cy.wrap(SOME_MODEL_NAME);
    });
  });
