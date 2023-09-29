import TestUtils from '../../support/TestUtils';
import { zincDatasetFixture } from '../../support/dataset/examples';

describe('Model version form (/models/new)', () => {
  before(() => {
    cy.on(
      'uncaught:exception',
      (err) => err.toString().includes('ResizeObserver') && false
    );
    
    cy.loginUser();
    cy.setupZincDataset();
  });

  const testModel = {
    modelName: 'asidjaisd',
    modelDescription: 'iajsdijasid',
    config: { name: 'ajisdjiasdjasjd' },
    modelVersionDescription: 'asdiasjd',
  };
  const testDatasetConfig = {
    name: zincDatasetFixture.name,
    targetColumns: [{ name: 'tpsa' }],
    featureColumns: [{ name: 'smiles' }, { name: 'mwt' }],
  };
  const visitPage = (suffix?: string) =>
    cy.visit('/models/new' + (suffix || ''));

  const clickNext = () => cy.get(`[data-testid="next"]`).click();

  const clickPrevious = () =>
    cy.get(`[data-testid="previous"]`).click({ timeout: 10000 });

  const fillModelForm = (
    modelFormData?: typeof testModel,
    datasetFormData?: typeof testDatasetConfig
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
      clickNext();
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

      clickNext();
    }
  };

  beforeEach(() => {
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
        cy.get('[data-testid="model-name"] input').clear({ force: true });
        cy.get('[data-testid="version-name"] input').clear({ force: true });
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
        cy.wrap(fillModelForm(testModel)).then(() => {
          clickPrevious().then(() => {
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
      });
    });

    describe('Features and Target inputs', () => {
      it('Is persisted across step transitions', () => {
        fillModelForm(testModel, testDatasetConfig);
        clickPrevious().then(() => {
          cy.get('[data-testid="dataset-target-column"]').contains(
            testDatasetConfig.targetColumns[0].name,
            { timeout: 10000 }
          );
          cy.wrap(
            testDatasetConfig.featureColumns.forEach((col) => {
              cy.get('[title="Feature Column"] span')
                .contains(col.name)
                .should('exist');
            })
          );

          cy.get('[data-testid="dataset-selector"] input').should(
            'have.value',
            testDatasetConfig.name
          );
        });
      });
      it('Validate required fields (dataset.name, dataset.targetColumns, dataset.featureColumns)', () => {
        fillModelForm(testModel);
        cy.get('#dataset-select').focus().blur();

        cy.get('[data-testid="dataset-selector"] label').should(
          'have.class',
          TestUtils.errorClass
        );
        clickNext();

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

        clickNext();
        cy.notificationShouldContain('Missing dataset target column selection');
      });
    });
  });
});
