import DatasetCreate from '@features/datasets/pages/DatasetCreate';
import { Box, CircularProgress } from '@mui/material';
import { Suspense } from 'react';
import { DefaultProviders } from '../support/DefaultProviders';
import TestUtils from '../support/TestUtils';
import { zincDatasetFixture } from '../support/dataset/examples';

describe('DatasetCreate.cy.tsx', () => {
  const MountedComponent = () => {
    return (
      <DefaultProviders>
        <Suspense
          fallback={
            <Box
              sx={{
                color: '#fff',
                width: '100vw',
                height: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <CircularProgress color="primary" />
            </Box>
          }
        >
          <DatasetCreate onCreate={() => {}} />
        </Suspense>
      </DefaultProviders>
    );
  }
  
  beforeEach(() => {
    cy.mount(<MountedComponent />)
  });

  it('Feedback missing required fields', () => {
    cy.get('button').contains('Save').click();
    // cy.wait(300);
    cy.get('#dataset-name-input')
      .parent({ timeout: 3000 })
      .should('have.class', TestUtils.errorClass);
    cy.get('#description-input')
      .siblings()
      .should('have.class', TestUtils.errorClass);
    cy.get('#dataset-upload')
      .should('have.class', 'invalid');


    cy.get('#dataset-upload').attachFile(zincDatasetFixture.file)
      .wait(300) //? Mocked "csv-metadata" request takes 200ms to resolve
      .then(() => {
        cy.get('button').contains('Save').click();
    
        cy.get('#dataset-name-input')
          .parent()
          .should('have.class', TestUtils.errorClass);
        cy.get('#description-input')
          .siblings()
          .should('have.class', TestUtils.errorClass);
    
        cy.get('#dataset-split-input')
          .parent()
          .should('have.class', TestUtils.errorClass);
      })
  });

  it('Shows required split column when split type is not random', () => {
    cy.get('#dataset-upload').attachFile(zincDatasetFixture.file);
    
    cy.get('#dataset-splittype-input', { timeout: 60000 })
      .click()
      .get('li')
      .contains('Scaffold')
      .click();
    cy.get('#dataset-split-column-input').should('exist');
    cy.get('button').contains('Save').click();
    cy.get('#dataset-split-column-input', { timeout: 2000 })
      .parent()
      .should('have.class', TestUtils.errorClass);
  });
});
