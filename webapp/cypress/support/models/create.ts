import { randomLowerCase } from '@utils';
import { SOME_MODEL_NAME } from '../constants';

export const modelFormData = (datasetName: string) => ({
  name: SOME_MODEL_NAME,
  modelDescription: 'cqlqrats',
  modelVersionDescription: 'fwscttrs',
  config: {
    name: 'test model version',
    dataset: {
      featureColumns: [
        {
          name: 'sepal_length',
          dataType: {
            domainKind: 'numeric',
            unit: 'mole',
          },
        },
        {
          name: 'sepal_width',
          dataType: {
            domainKind: 'numeric',
            unit: 'mole',
          },
        },
      ],
      featurizers: [],
      targetColumns: [
        {
          type: 'output',
          name: 'large_petal_length',
          dataType: {
            domainKind: 'categorical',
            classes: {
              '0': 0,
              '1': 1,
            },
          },
          forwardArgs: {
            '': '$Linear-5',
          },
          outModule: 'Linear-5',
          columnType: 'binary',
          lossFn: 'torch.nn.BCEWithLogitsLoss',
        },
      ],
      name: datasetName,
    },
    spec: {
      layers: [
        {
          type: 'fleet.model_builder.layers.Concat',
          forwardArgs: {
            xs: ['$sepal_length', '$sepal_width'],
          },
          constructorArgs: {
            dim: 1,
          },
          name: 'Concat-0',
        },
        {
          type: 'torch.nn.Linear',
          forwardArgs: {
            input: '$Concat-0',
          },
          constructorArgs: {
            in_features: 2,
            out_features: 16,
            bias: true,
          },
          name: 'Linear-1',
        },
        {
          type: 'torch.nn.ReLU',
          forwardArgs: {
            input: '$Linear-1',
          },
          constructorArgs: {
            inplace: false,
          },
          name: 'ReLU-2',
        },
        {
          type: 'torch.nn.Linear',
          forwardArgs: {
            input: '$ReLU-2',
          },
          constructorArgs: {
            in_features: 16,
            out_features: 16,
            bias: true,
          },
          name: 'Linear-3',
        },
        {
          type: 'torch.nn.ReLU',
          forwardArgs: {
            input: '$Linear-3',
          },
          constructorArgs: {
            inplace: false,
          },
          name: 'ReLU-4',
        },
        {
          type: 'torch.nn.Linear',
          forwardArgs: {
            input: '$ReLU-4',
          },
          constructorArgs: {
            in_features: 16,
            out_features: 1,
            bias: true,
          },
          name: 'Linear-5',
        },
      ],
    },
  },
});

const modelExists = (name: string): Cypress.Chainable<boolean> =>
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

const createModelDirectly = (modelFormData: any) =>
  cy.getCurrentAuthString().then((authorization) =>
    cy
      .request({
        method: 'POST',
        url: `http://localhost/api/v1/models`,
        headers: {
          authorization,
        },
        body: modelFormData,
      })
      .then((response) => {
        expect(response?.status).to.eq(200);
      })
  );

export const setupSomeModel = () =>
  modelExists(SOME_MODEL_NAME).then((exists) => {
    return cy.setupIrisDatset().then((fixture) => {
      const formData = modelFormData(fixture.name);

      if (exists) {
        return cy.wrap(formData);
      }

      createModelDirectly(formData);
      return cy.wrap(formData);
    });
  });
