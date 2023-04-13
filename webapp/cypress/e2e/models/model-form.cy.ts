import { ModelSchema } from '@app/rtk/generated/models';
import TestUtils from '../../support/TestUtils';

describe('Model version form (/models/new)', () => {
  const testModel = {
    modelName: 'asidjaisd',
    modelDescription: 'iajsdijasid',
    config: { name: 'ajisdjiasdjasjd' },
    modelVersionDescription: 'asdiasjd',
  };
  const testDatasetConfig = {
    name: 'ZincExtra',
    targetColumns: [{ name: 'tpsa' }],
    featureColumns: [{ name: 'smiles' }, { name: 'mwt' }],
  };
  const visitPage = (suffix?: string) =>
    cy.visit('/models/new' + (suffix || ''));

  const clickNext = (stepperRootChildIndex: number) =>
    cy
      .get(
        `.MuiStepper-root > :nth-child(${stepperRootChildIndex}) [data-testid="next"]`
      )
      .click();

  const clickPrevious = (stepperRootChildIndex: number) =>
    cy
      .get(
        `.MuiStepper-root > :nth-child(${stepperRootChildIndex}) [data-testid="previous"]`
      )
      .click();

  const fillModelForm = (
    modelFormData?: typeof testModel,
    datasetFormData?: typeof testDatasetConfig,
    modelConfig?: ModelSchema
  ) => {
    if (modelFormData) {
      cy.get('[data-testid="model-name"] input')
        .type(modelFormData.modelName)
        .wait(500)
        .type('{enter}');
      cy.get('[data-testid="model-description"] input').type(
        modelFormData.modelDescription
      );
      cy.get('[data-testid="version-name"] input').type(
        modelFormData.config.name
      );
      cy.get('[data-testid="version-description"] textarea').type(
        modelFormData.modelVersionDescription
      );
      clickNext(1);
    }
    if (datasetFormData) {
      cy.get('#dataset-select')
        .click()
        .type(datasetFormData.name || '')
        .get('li[role="option"]')
        .first()
        .click();

      cy.get('#target-col').click();
      cy.get('div').contains(datasetFormData.targetColumns[0].name).click();
      datasetFormData.featureColumns.forEach((col) => {
        cy.get('#feature-cols').click();
        cy.get('div').contains(col.name).click();
      });
      // TODO: fill data

      clickNext(3);
    }
    if (modelConfig) {
      // TODO: fill data
    }
  };

  beforeEach(() => {
    cy.loginSuper();
    visitPage();
  });

  describe('Adding model and first model version simultenously', () => {
    describe('Model description inputs', () => {
      it('Model form starts with empty model name and model description', () => {
        cy.get('[data-testid="model-name"] input').should('have.value', '');
      });
      it('Model name random generation works', () => {
        cy.get('[data-testid="model-name"] input').invoke(
          'text',
          (previousText: string) => {
            cy.get('[data-testid="random-model-name"]').click();
            cy.get('[data-testid="model-name"] input').should(
              'not.have.value',
              previousText
            );
          }
        );
      });
      it('Validation on required fields (model name and model version name)', () => {
        cy.get('[data-testid="model-name"] input').clear();
        cy.get('[data-testid="version-name"] input').clear();
        cy.get('[data-testid="next"]').click();
        cy.notificationShouldContain('Missing');
        cy.get('[data-testid="model-name"] label').should(
          'have.class',
          TestUtils.errorClass
        );
        cy.get('[data-testid="version-name"] label').should(
          'have.class',
          TestUtils.errorClass
        );
      });
      it('Is persisted across step transitions', () => {
        fillModelForm(testModel);
        clickPrevious(3);
        cy.get('[data-testid="model-description"] input').should(
          'have.value',
          testModel.modelDescription
        );
        cy.get('[data-testid="version-name"] input').should(
          'have.value',
          testModel.config.name
        );
        cy.get('[data-testid="version-description"] textarea').should(
          'have.value',
          testModel.modelVersionDescription
        );

        cy.get('[data-testid="model-name"] input').should(
          'have.value',
          testModel.modelName
        );
      });
    });
    describe('Features and Target inputs', () => {
      it('Reads dataset from querystring datasetId (if present) to fill the input', () => {
        // TODO: mock dataset fetch call
        const knownDataset = { datasetId: 156, datasetName: 'ZincExtra' };
        visitPage(`?datasetId=${knownDataset.datasetId}`);
        fillModelForm(testModel);
        cy.get('#dataset-select').should(
          'have.value',
          knownDataset.datasetName
        );
      });
      it('Is persisted across step transitions', () => {
        fillModelForm(testModel, testDatasetConfig);
        clickPrevious(5);
        cy.get('#target-col').should(
          'have.value',
          testDatasetConfig.targetColumns[0].name
        );

        testDatasetConfig.featureColumns.forEach((col) => {
          cy.get('[title="Feature Column"] span')
            .contains(col.name)
            .should('exist');
        });

        cy.get('#dataset-select').should('have.value', testDatasetConfig.name);
      });
      it('Validate required fields (dataset.name, dataset.targetColumns, dataset.featureColumns)', () => {
        fillModelForm(testModel);
        cy.get('#dataset-select').focus().blur();

        cy.get('[data-testid="dataset-selector"] label').should(
          'have.class',
          TestUtils.errorClass
        );
        clickNext(3);

        cy.notificationShouldContain('Missing dataset name');

        cy.get('#dataset-select')
          .click()
          .type(testDatasetConfig.name || '')
          .get('li[role="option"]')
          .first()
          .click();

        cy.get('#target-col').focus().blur();
        cy.get('[data-testid="dataset-target-column"] label').should(
          'have.class',
          TestUtils.errorClass
        );
        cy.get('#feature-cols').focus().blur();
        cy.get('[data-testid="dataset-feature-columns"] label').should(
          'have.class',
          TestUtils.errorClass
        );

        clickNext(3);
        cy.notificationShouldContain('Missing dataset target column selection');
      });
    });
    describe('Neural Net Architecture editor (model editor/buillder/compiler)', () => {
      it.skip('Suggestions are visible and apparently fixable to the user', () => {});
      it('Allows connecting 2 nodes on key press', () => {
        throw 'error';
      });
    });
  });

  describe('Adding model version to existing model', () => {
    it.skip('Starts with model name and model descriptions filled with existing model data and disabled', () => {});
    it.skip('Starts with dataset name, targetColumns and featureColumns filled with existing model data (allowing to change it)', () => {});
  });
});
