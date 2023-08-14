/// <reference types="cypress" />

import { ModelSchema } from 'model-compiler/src/interfaces/model-editor';
import { iterateTopologically } from 'model-compiler/src/utils';
import { substrAfterLast } from '@utils';
import {
  MODEL_SMILES_CATEGORICAL_NAME,
  MODEL_SMILES_NUMERIC_NAME,
} from '../constants';

const SCHEMA_PATH = Cypress.env('SCHEMA_PATH');

const API_BASE_URL = Cypress.env('API_BASE_URL');

type ComponentTypeByDataId = {
  [key: string]: { type: string; filled: Record<string, boolean> };
};

export const flowDropSelector =
  'div[class="react-flow__pane react-flow__container"]';

const dragComponent = (component: string, x: number, y: number): void => {
  const sourceSelector = 'div[draggable="true"]';
  cy.get(sourceSelector).contains(component).drag(flowDropSelector, x, y);
};
const generatePositions = (
  n: number,
  xOffset: number,
  yOffset: number,
  maxWidth: number
): { x: number; y: number }[] => {
  let yidx = 0;
  let xidx = 0;
  const positions: { x: number; y: number }[] = [];
  for (let i = 0; i < n; i++) {
    positions.push({
      x: xidx * xOffset + 150,
      y: (yidx + 1) * yOffset,
    });
    if (xidx * xOffset >= maxWidth) {
      xidx = 0;
      yidx += 1;
    } else {
      xidx += 1;
    }
  }
  return positions;
};

/**
 * Drags the needed components to assemble `config` into the editor and
 * returns of itself with the layers and featurizers names updated so it matches
 * the HTML id attributes in the graph editor
 *
 *  It uses the fact that editor node ids are incremented as nodes are added
 *
 * @param {ModelSchema} config
 * @returns {ModelSchema}
 */
export const dragComponentsAndMapConfig = (
  config: ModelSchema
): ModelSchema => {
  const layers = config.spec.layers || [];
  const featurizers = config.dataset.featurizers || [];
  const total = (layers?.length || 0) + (featurizers?.length || 0);
  const xoffset = 150;
  const yoffset = 50;
  const maxwidth = 700;
  const positions = generatePositions(total, xoffset, yoffset, maxwidth);
  let i = 0;
  const newLayers: ModelSchema['spec']['layers'] = [];
  const newFeaturizers: ModelSchema['dataset']['featurizers'] = [];
  iterateTopologically(config, (node, type) => {
    cy.log('Dragging component', node, type);
    if (['input', 'output'].includes(type)) return;
    const position = positions[i];
    const componentName = substrAfterLast(node.type || '', '.');
    dragComponent(componentName, position.x, position.y);
    if (type === 'layer') {
      //@ts-ignore
      const layer = {
        ...node,
        type: `${node.type}-${i}`,
      };
      //@ts-ignore
      newLayers.push(layer);
    } else if (type === 'featurizer') {
      //@ts-ignore
      const featurizer = {
        ...node,
        type: `${node.type}-${i}`,
      };
      //@ts-ignore
      newFeaturizers.push(featurizer);
    }
    i += 1;
  });
  const newSchema = {
    ...config,
    dataset: {
      ...config.dataset,
      featurizers: newFeaturizers,
    },
    spec: {
      layers: newLayers,
    },
  };
  return newSchema;
};
export const dragComponents = (componentNames: string[]) => {
  cy.get('[data-testid="TurnLeftIcon"]').click();
  const xoffset = 150;
  const yoffset = 80;
  const maxwidth = 150;
  const positions = generatePositions(
    componentNames.length,
    xoffset,
    yoffset,
    maxwidth
  );
  componentNames.forEach((component, i) => {
    const { x, y } = positions[i];
    dragComponent(component, x, y);
  });
};

export const fillDatasetInfo = ({
  modelName,
  targetCol,
  featureCols,
}: {
  modelName: string;
  targetCol: string;
  featureCols: string[];
}) => {
  cy.get('#model-name').clear();
  cy.get('#model-name')
    .type(modelName)
    .wait(1000)
    .get('li[role="option"]')
    .first()
    .click();
  cy.get('#dataset-select').click().get('li[role="option"]').first().click();
  cy.get('#target-col').click();
  cy.get('div').contains(targetCol).click();
  featureCols.forEach((col) => {
    cy.get('#feature-cols').click();
    cy.get('div').contains(col).click();
  });
};
//? Not being used currently
export const createModelFixture = (
  modelData: Parameters<typeof fillDatasetInfo>[0]
) => {
  cy.visit('/models/new');
  fillDatasetInfo(modelData);
  cy.get('button').contains('NEXT').click();
  cy.buildYamlModel(SCHEMA_PATH + 'yaml/small_regressor_schema.yaml');
};

export const connectAndFill = ({
  argValues = {},
  sourceId,
  targetId,
  sourceHandle,
  targetHandle,
  componentTypeByDataId,
}: {
  sourceId: string;
  argValues?: { [key: string]: string | boolean };
  targetId: string;
  sourceHandle?: string;
  targetHandle?: string;
  componentTypeByDataId: ComponentTypeByDataId;
}) => {
  Cypress.log({
    name: 'connectAndFill',
    displayName: 'Connecting 2 components',
    message: `Connection ${sourceId}${
      sourceHandle ? '.' + sourceHandle : ''
    } to node ${targetId}${targetHandle ? '.' + targetHandle : ''}`,
  });
  cy.get(`div[data-id="${sourceId}"]`)
    .find(
      `.react-flow__handle.source${generateEspecificHandleSelector(
        sourceHandle || ''
      )}`
    )
    .trigger('mousemove', { force: true })
    .click({ force: true })
    .trigger('mousedown', {
      button: 0,
      force: true,
      waitForAnimations: true,
      log: true,
    });
  cy.get(`div[data-id="${targetId}"]`)
    .find(
      `.react-flow__handle.target${generateEspecificHandleSelector(
        targetHandle || ''
      )}`
    )
    .last()
    .trigger('mousemove', { force: true })
    .wait(100)
    .trigger('mouseup', { force: true })
    .click({ force: true })
    .wait(100);
  const openArgs = (dataId: string) => {
    cy.get(`div[data-id="${dataId}"]`)
      .find('svg[data-testid="OpenInFullRoundedIcon"]')
      .click();
  };
  const closeArgs = (dataId: string) => {
    cy.get(`div[data-id="${dataId}"]`)
      .find('svg[data-testid="MinimizeRoundedIcon"]')
      .click()
      .wait(1000);
  };

  Object.entries(argValues).forEach(([key, value]) => {
    openArgs(targetId);

    if (typeof value === 'boolean' && value) {
      cy.get(`div[data-id="${targetId}"]`)
        .find(`#${key}-input`)
        .click({ force: true });
    } else if (typeof value === 'string' && value) {
      cy.get(`div[data-id="${targetId}"]`)
        .find(`#${key}-input`)
        .focus()
        .type('{selectall}')
        .type(value, { force: true });
    }
    closeArgs(targetId);
  });

  if (targetId in componentTypeByDataId)
    componentTypeByDataId[targetId].filled[targetHandle || 'input'] = true;
};

const generateEspecificHandleSelector = (handleName: string) => {
  if (!handleName) return '';
  return `[data-handleid="${handleName}"]`;
};

export const findNotFilled = (
  type: string,
  components: ComponentTypeByDataId,
  targetHandle: string = 'input'
) => {
  const comp = Object.entries(components).find(
    ([_key, value]) =>
      value.type.startsWith(type) &&
      (type === 'Concat' || !value.filled[targetHandle])
  );
  const filled = Object.entries(components).find(
    ([_key, value]) => value.type.startsWith(type) && value.filled[targetHandle]
  );

  if (!comp) {
    throw new Error(
      'no component found ' + type + '\n' + JSON.stringify(filled, null, 2)
    );
  }

  return comp[0];
};

export const deleteTestModelsIfExist = () => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/models/*`,
  }).as('getModels');
  cy.visit('/models');
  cy.wait('@getModels');
  cy.get('tbody').then(($tbody) => {
    if ($tbody.find('a').length) {
      cy.get('tbody a').each(($link) => {
        if (
          [MODEL_SMILES_NUMERIC_NAME, MODEL_SMILES_CATEGORICAL_NAME].includes(
            $link.text()
          )
        ) {
          cy.get('a').contains($link.text()).click();
          cy.wait(1000);
          cy.get('button').contains('Delete').click().wait(1000);
        }
      });
    }
  });
};

export const deleteModelIfExist = (modelName: string) => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/models/*`,
  }).as('getModels');
  cy.visit('/models');
  cy.wait('@getModels');
  cy.get('tbody').then(($tbody) => {
    if ($tbody.find('a').length) {
      cy.get('tbody a').each(($link) => {
        if ($link.text() === modelName) {
          cy.get('a').contains($link.text()).click();
          cy.wait(1000);
          cy.get('button').contains('Delete').click().wait(1000);
        }
      });
    }
  });
};
