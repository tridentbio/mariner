import { ModelCreate } from "@app/rtk/generated/models"
import { store } from "@app/store"
import DataPreprocessingInput from "@components/organisms/ModelBuilder/DataPreprocessingInput"
import { ModelBuilderContextProvider } from "@components/organisms/ModelBuilder/hooks/useModelBuilder"
import { SimpleColumnConfig, StepValue } from "@components/organisms/ModelBuilder/types"
import { getColumnConfigTestId, getStepValueLabelData } from "@components/organisms/ModelBuilder/utils"
import { schema } from '@features/models/pages/ModelCreateV2'
import { yupResolver } from "@hookform/resolvers/yup"
import { ThemeProvider } from "@mui/system"
import { FormProvider, NonUndefined, useForm } from "react-hook-form"
import { Provider } from "react-redux"
import { theme } from "theme"

describe('DataPreprocessingInput.cy.tsx', () => {
  const value = {
    name: 'v1',
    config: {
      dataset: {
        name: 'Test dataset',
        strategy: 'pipeline',
        featureColumns: [
          {
            name: 'Smiles Column 2',
            dataType: { domainKind: 'smiles' },
            featurizers: [],
            transforms: [],
          },
          {
            name: 'DNA Column 2',
            dataType: { domainKind: 'dna' },
            featurizers: [],
            transforms: [],
          },
          {
            name: 'RNA Column 2',
            dataType: { domainKind: 'rna' },
            featurizers: [],
            transforms: [],
          },
          {
            name: 'Protein Column 2',
            dataType: { domainKind: 'protein' },
            featurizers: [],
            transforms: [],
          },
        ],
        targetColumns: [
          {
            name: 'Numerical Column 1',
            dataType: { domainKind: 'numeric', unit: 'mole' },
            featurizers: [],
            transforms: [],
          },
        ],
      },
      name: 'Test model',
      framework: 'sklearn',
      spec: {
        model: {
          fitArgs: {},
          type: 'sklearn.ensemble.RandomForestRegressor',
        },
      },
    },
  } as ModelCreate

  const MountedComponent = () => {
    const methods = useForm<ModelCreate>({
      defaultValues: value,
      mode: 'all',
      criteriaMode: 'all',
      reValidateMode: 'onChange',
      resolver: yupResolver(schema),
    });

    return (
      <Provider store={store}>
        <ThemeProvider theme={theme}>
          <FormProvider {...methods}>
            <ModelBuilderContextProvider>
              <DataPreprocessingInput
                value={value?.config?.dataset?.featureColumns as SimpleColumnConfig[] || []}
                type="featureColumns"
              />
              <DataPreprocessingInput
                value={value?.config?.dataset?.targetColumns as SimpleColumnConfig[] || []}
                type="targetColumns"
              />
            </ModelBuilderContextProvider>
          </FormProvider>
        </ThemeProvider>
      </Provider>
    )
  }

  const FEATURIZERS_DICTIONARY: {
    [dataType in NonUndefined<SimpleColumnConfig['dataType']['domainKind']>]?: StepValue['type'][]
  } = {
    dna: ['fleet.model_builder.featurizers.DNASequenceFeaturizer'],
    rna: ['fleet.model_builder.featurizers.RNASequenceFeaturizer'],
    categorical: ['sklearn.preprocessing.LabelEncoder', 'sklearn.preprocessing.OneHotEncoder'],
    protein: ['fleet.model_builder.featurizers.ProteinSequenceFeaturizer'],
    smiles: ['molfeat.trans.fp.FPVecFilteredTransformer'],
  }

  it('should filter compatible featurizers on preprocessing step select input', () => {
    cy.mount(<MountedComponent />)
    
    const cols = (value?.config?.dataset?.featureColumns as SimpleColumnConfig[])
      .concat(value?.config?.dataset?.targetColumns as SimpleColumnConfig[])

    cols.forEach(col => {
      const colId = getColumnConfigTestId(col! as SimpleColumnConfig)
      const colAccordion = cy.get(`[data-testid="${colId}-accordion"]`)
      const colDomainType = col.dataType.domainKind as keyof typeof FEATURIZERS_DICTIONARY

      colAccordion.click()

      cy.get("body").then($body => {
        const colHasFeaturizer = $body.find(`[data-testid="${colId}-featurizer-label"]`).length > 0
        
        if(colHasFeaturizer) {
          cy.get(`[data-testid="${colId}-featurizer-0"] .step-select`).click()

          FEATURIZERS_DICTIONARY[colDomainType]
            ?.forEach(featurizerType => {
              const stepLabelData = getStepValueLabelData(featurizerType)

              //? Validate featurizer valid options being listed
              cy.get('li[role="option"]').should('contain.text', stepLabelData?.class)
              
              if(colDomainType == 'smiles') cy.get('ul[role="listbox"]').find('li').should('have.length', 1)

              //? Validate featurizer invalid options not being listed
              Object.entries(FEATURIZERS_DICTIONARY).forEach(([domainType, featurizers]) => {
                if(domainType !== colDomainType) {
                  featurizers?.forEach(featurizer => {
                    cy.get('li[role="option"]').should('not.contain.text', getStepValueLabelData(featurizer)?.class)
                  })
                }
              })
            })
        }
      })
    })
  })
})