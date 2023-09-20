/// <reference types="cypress" />

import { ColumnConfig, DatasetConfig, ModelCreate, SklearnModelSchema, TorchModelSpec } from '@app/rtk/generated/models';
import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import { getColumnConfigTestId, getStepValueLabelData } from '@components/organisms/ModelBuilder/utils';
import { DeepPartial } from '@reduxjs/toolkit';
import { NodeType } from '@model-compiler/src/interfaces/torch-model-editor';
import {
  extendSpecWithTargetForwardArgs,
  iterateTopologically,
  unwrapDollar,
} from 'model-compiler/src/utils';
import { isArray, randomLowerCase } from 'utils';
import { parse } from 'yaml';
import { dragComponentsAndMapConfig, flowDropSelector } from './common';

type SklearnColumnFeaturizer = NonNullable<DatasetConfig['featurizers']>[0]
type SklearnColumnTransform = NonNullable<DatasetConfig['transforms']>[0]

const API_BASE_URL = Cypress.env('API_BASE_URL');

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

export const autoFixSuggestions = () =>
  cy.getWithoutThrow('[data-testid="AutoFixHighOutlinedIcon"]').then(($els) => {
    if ($els.length === 1)
      cy.get('[data-testid="AutoFixHighOutlinedIcon"]')
        .first()
        .click({ force: true });
    else if ($els.length !== 0)
      cy.get('li').contains('Fix all').click({ force: true });
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

export const fillModelDescriptionStepForm = (
  modelCreate: DeepPartial<ModelCreate>
) => {
  cy.intercept({
    method: 'GET',
    url: `${API_BASE_URL}/api/v1/models/*`,
  }).as('getModels');
  
  cy.visit('/models/new');

  cy.wait('@getModels').then(({ response }) => {
    // Fill model name
    cy.get('[data-testid="model-name"] input', {timeout: 10000})
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
  })
};

export const fillDatasetCols = (cols: (ColumnConfig | SimpleColumnConfig)[], colInputSelector: string) => {
  cols.map(col => {
    const colId = getColumnConfigTestId(col! as (ColumnConfig | SimpleColumnConfig))
  
    cy.get(colInputSelector).click();
    cy.get(`li[data-testid="${colId}"`)
      .click();
  })
}

export const buildModel = (
  modelCreate: DeepPartial<ModelCreate>,
  params: {
    /** default `true` */
    successfullRequestRequired?: boolean;
    /** default `true` */
    applySuggestions?: boolean;
    /** default `true` */
    submitModelRequest?: boolean;
  } = {}
) => {
  params.successfullRequestRequired = params.successfullRequestRequired ?? true;
  params.applySuggestions = params.applySuggestions ?? true;
  params.submitModelRequest = params.submitModelRequest ?? true;

  cy.once('uncaught:exception', () => false);
  cy.visit('/models/new');

  fillModelDescriptionStepForm(modelCreate);

  cy.get('button').contains('NEXT').click();

  cy.get('#dataset-select')
    .click()
    .type(modelCreate.config?.dataset?.name || '')
    .get('li[role="option"]')
    .first()
    .click();

  const featureCols = modelCreate.config?.dataset?.featureColumns || [];
  const targetCols = modelCreate.config?.dataset?.targetColumns || [];

  fillDatasetCols(
    (targetCols as (SimpleColumnConfig[] | ColumnConfig[])) ?? [],
    '#target-col'
  );
  
  fillDatasetCols(
    (featureCols as (SimpleColumnConfig[] | ColumnConfig[])) ?? [],
    '#feature-cols'
  );

  cy.get('#framework-selector').click()    
  cy.get(`[role="option"][data-value="${modelCreate.config?.framework}"]`).click()

  if(modelCreate.config?.framework == 'torch') {
    buildFlowSchema(modelCreate, params);
  } else {
    cy.get('button').contains('NEXT').click();

    buildSklearnPreprocessingForm(modelCreate, featureCols as SimpleColumnConfig[], targetCols as SimpleColumnConfig[]);
  }

  if (params.submitModelRequest) {
    cy.intercept({
      method: 'POST',
      url: `${API_BASE_URL}/api/v1/models/check-config`,
    }).as('checkConfig');
    cy.intercept({
      method: 'POST',
      url: `${API_BASE_URL}/api/v1/models`,
    }).as('createModel');

    cy.get('button').contains('CREATE').click();

    if(modelCreate.config?.framework == 'torch') {
      cy.wait('@checkConfig', {responseTimeout: 60000}).then(({ response }) => {
        expect(response?.statusCode).to.eq(200);
        if (params.successfullRequestRequired) {
          expect(Boolean(response?.body.stackTrace)).to.eq(false);
        }
      });
    }

    if (params.successfullRequestRequired)
      cy.wait('@createModel').then(({ response }) => {
        expect(response?.statusCode).to.eq(200);
      });
  }
};

const buildSklearnPreprocessingForm = (
  modelCreate: DeepPartial<ModelCreate>,
  featureCols: SimpleColumnConfig[],
  targetCols: SimpleColumnConfig[]
) => {
  const sklearnDataset = modelCreate.config?.dataset as DatasetConfig
  const cols = featureCols.concat(targetCols)

  cols.map((col, colIndex) => {
    const colId = getColumnConfigTestId(col! as SimpleColumnConfig)

    const colAccordion = cy.get(`[data-testid="${colId}-accordion"]`)
    colAccordion.click()

    let processors: {
      featurizers: SklearnColumnFeaturizer[]
      transforms: SklearnColumnTransform[]
    } = {
      featurizers: [],
      transforms: []
    }

    let columnTransforms: SklearnColumnTransform[] = sklearnDataset.transforms?.filter(
      transform => filterColumnRelatedSteps(col.name, transform)
    ) || []

    processors.transforms = columnTransforms

    //? Fill column steps
    cy.get("body").then($body => {
      const colHasFeaturizer = $body.find(`[data-testid="${colId}-featurizer-label"]`).length > 0

      //? Fill column featurizer (if exists)
      if (colHasFeaturizer) {
        const foundFeaturizer = sklearnDataset.featurizers?.find(featurizer => filterColumnRelatedSteps(col.name, featurizer))

        if(!foundFeaturizer) throw new Error(`No featurizer found for column ${col?.name}`)

        processors.featurizers = [foundFeaturizer]
      }

      Object.entries(processors).forEach(([processorType, steps]) => {
        steps.forEach((step, stepIndex) => {
          const stepName = processorType == 'featurizers' ? 'featurizer' : 'transform'
          const stepLabelData = getStepValueLabelData(step.type)

          if(stepName == 'transform') colAccordion.find('button').contains('ADD').click()
  
          cy.get(`[data-testid="${colId}-${stepName}-${stepIndex}"] .step-select`)
            .click()
            .type(stepLabelData?.class || '')
            .get('li[role="option"]')
            .first()
            .click();
  
          //? Fill step constructorArgs
          if('constructorArgs' in step && step.constructorArgs) {
            const actionBtnId = `${colId}-${stepName}-${stepIndex}-action-btn`
  
            cy.get(`[data-testid="${actionBtnId}"]`).click()
  
            const args: {[key: string]: any} = step.constructorArgs || {}
  
            buildStepConstructorArgs(`${colId}-${stepName}-${stepIndex}`, args)
          }
        })
      })
    });
  })

  cy.get('button').contains('NEXT').click();

  const modelSchema = modelCreate.config?.spec as SklearnModelSchema
  const modelLabel = getStepValueLabelData(modelSchema.model.type)

  cy.get(`[data-testid="sklearn-model-select"] .step-select`)
    .click()
    .type(modelLabel?.class || '')
    .get('li[role="option"]')
    .first()
    .click();

  if('constructorArgs' in modelSchema.model && modelSchema.model.constructorArgs) {
    const args: {[key: string]: any} = modelSchema.model.constructorArgs || {}
    buildStepConstructorArgs(`sklearn-model-select`, args)
  }
}

const filterColumnRelatedSteps = (colName: string, processor: (NonNullable<DatasetConfig['featurizers']> | NonNullable<DatasetConfig['transforms']>)[0]) => {
  const referenceCols = (processor?.forwardArgs as {[key: string]: any})['X']

  if(isArray(referenceCols)) {
    const unwrappedColNameList = referenceCols.map(colName => unwrapDollar(colName))

    return colName && unwrappedColNameList.includes(colName)
  } else {
    return unwrapDollar(referenceCols) == colName
  }
}

const buildStepConstructorArgs = (stepId: string, args: {[key: string]: any}) => {
  type ArgInputType = 'string' | 'number' | 'boolean' | 'options'

  Object.keys(args).map(arg => {
    const argInput = cy.get(`[data-testid="${stepId}"] #arg-option-${arg}`)

    argInput.then($argInput => {
      const argInputType = $argInput.attr('data-argtype') as ArgInputType

      switch(argInputType) {
        case 'options':
          argInput.click()    
          cy.get(`[role="option"][data-value="${args[arg]}"]`).click({force: true})
          break
        case 'boolean':
          !!args[arg] ? argInput.check({force: true}) : argInput.uncheck({force: true})
          break
        default:
          argInput
            .click()
            .clear()
            .type(args[arg])
      }
    })
  })
}

const buildFlowSchema = (
  modelCreate: DeepPartial<ModelCreate>,
  params: Parameters<typeof buildModel>[1]
) => {
  cy.get('button').contains('NEXT').click();

  cy.get(flowDropSelector).trigger('wheel', {
    deltaY: 0,
    wheelDeltaX: 0,
    wheelDeltaY: 2500,
    bubbles: true,
  });

  const mod = extendSpecWithTargetForwardArgs(
    modelCreate.config as TorchModelSpec
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

  cy.then(() => {
    iterateTopologically(config, (node, type) => {
      if (['input', 'output'].includes(type)) return;
      const args = 'constructorArgs' in node ? node.constructorArgs : {};
      if (!args || objIsEmpty(args)) return;

      Object.entries(args).forEach(([key, value]) => {
        if (params?.applySuggestions) autoFixSuggestions();

        const parsedEdgeName = parseEdgeName(node.type!)[0];

        const element = cy.get(`[data-id="${parsedEdgeName}"]`).first();

        const isBool = !['number', 'string'].includes(typeof value);

        if (!isBool) {
          const curElement = element
            .find(`[data-testid="${parsedEdgeName}-${key}"]`)
            .find('input');
          curElement.clear().type(value as string);
        } else {
          element.get(`[id="${key}"]`).then((curElement) => {
            if (Boolean(curElement.prop('checked')) !== value)
              curElement.trigger('click');
          });
        }
      });
    });
  });
};

export const buildYamlModel = (
  yamlPath: string,
  dataset: string | null = null,
  buildParams: Parameters<typeof buildModel>[1],
  modelName = randomName()
) => (
  cy.readFile(yamlPath).then((yamlStr) => {
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
      buildParams
    );
  })
)
