import { DatasetFormData } from '../../../support/dataset/create';
import { autoFixSuggestions } from '../../../support/models/build-model';

const SCHEMA_PATH = Cypress.env('SCHEMA_PATH');

describe('/models/new - Suggestions', () => {
  let zincDatasetFixture: DatasetFormData | null = null;

  const applyLayout = (layout: 'vertical' | 'horizontal' = 'horizontal') => {
    cy.get(`div[aria-label="Apply auto ${layout} layout"] button`).click();
    cy.get('button[title="fit view"]').click();
  };

  before(() => {
    cy.on(
      'uncaught:exception',
      (err) => err.toString().includes('ResizeObserver') && false
    );

    cy.loginSuper();

    cy.setupZincDataset().then((zinc) => {
      zincDatasetFixture = zinc;
    });
  });

  beforeEach(() => {
    cy.loginSuper();
    cy.visit('/models/new');
  });

  it('Applies LinearLinear validation suggestion', () => {
    cy.buildYamlModel(
      SCHEMA_PATH + '/yaml/linear_linear_validation_schema.yaml',
      zincDatasetFixture!.name,
      {
        applySuggestions: false,
        submitModelRequest: false,
      }
    );

    autoFixSuggestions();
    applyLayout();

    const generatedNonLinearNodesNames = [
      {
        name: 'Linear-3-Linear-2-ReLu',
        refersTo: 'Linear-2',
        pointsTo: 'Linear-3',
      },
      {
        name: 'Linear-2-Linear-1-ReLu',
        refersTo: 'Linear-1',
        pointsTo: 'Linear-2',
      },
      {
        name: 'Linear-1-Linear-0-ReLu',
        refersTo: 'Linear-0',
        pointsTo: 'Linear-1',
      },
    ];

    generatedNonLinearNodesNames.forEach((nonLinearNode) => {
      cy.get(`[data-id="${nonLinearNode.name}"]`).should('exist');

      //? Edges
      cy.get(`[id="${nonLinearNode.pointsTo}-${nonLinearNode.name}"]`).should(
        'exist'
      );
      cy.get(`[id="${nonLinearNode.name}-${nonLinearNode.refersTo}"]`).should(
        'exist'
      );
    });
  });
});
