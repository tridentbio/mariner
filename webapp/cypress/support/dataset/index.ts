import createDataset, {
  DatasetFormData,
  setupIrisDatset,
  setupZincDataset,
} from './create';

Cypress.Commands.add('createDataset', createDataset);
Cypress.Commands.add('setupIrisDatset', setupIrisDatset);
Cypress.Commands.add('setupZincDataset', setupZincDataset);

declare global {
  namespace Cypress {
    interface Chainable {
      createDataset(dataset: DatasetFormData): Cypress.Chainable<string>;
      setupIrisDatset(): Cypress.Chainable<DatasetFormData>;
      setupZincDataset(): Cypress.Chainable<DatasetFormData>;
    }
  }
}
