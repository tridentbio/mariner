import { buildNumSmilesModel } from './smiles-num';
import { buildCategoricalSmilesModel } from './smiles-categorical';

Cypress.Commands.add('buildNumSmilesModel', buildNumSmilesModel);
Cypress.Commands.add(
  'buildCategoricalSmilesModel',
  buildCategoricalSmilesModel
);

declare global {
  namespace Cypress {
    interface Chainable {
      buildNumSmilesModel(featureCols: string[], targetCol: string): Chainable;
      buildCategoricalSmilesModel(
        featureCols: string[],
        targetCol: string
      ): Chainable;
    }
  }
}
