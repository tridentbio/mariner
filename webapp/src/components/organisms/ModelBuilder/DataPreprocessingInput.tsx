import { Text } from '@components/molecules/Text';
import useModelOptions, {
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { Box } from '@mui/material';
import ColumnsPipelineInput from './ColumnsPipelineInput';
import {
  DatasetConfigPreprocessing,
  SimpleColumnConfig,
  StepValue,
} from './types';
import { ComponentOption } from '@app/rtk/generated/models';
export interface DataPreprocessingInputProps {
  value?: DatasetConfigPreprocessing;
}
const DataPreprocessingInput = ({ value }: DataPreprocessingInputProps) => {
  const options = useModelOptions();
  const { featureColumns, targetColumns } = value || {
    featureColumns: [],
    targetColumns: [],
  };

  const applyCompatibilityAttributeMock = (
    opt: ComponentOption
  ): ComponentOption => {
    const option: ComponentOption = Object.assign({}, opt);

    const categoricalFeaturizerTypes: StepValue['type'][] = [
      'sklearn.preprocessing.OneHotEncoder',
      'sklearn.preprocessing.LabelEncoder',
    ];

    option.compatibleWith = {
      domains:
        (option.classPath as StepValue['type']) ===
        'molfeat.trans.fp.FPVecFilteredTransformer'
          ? ['smiles']
          : (option.classPath as StepValue['type']) ===
            'fleet.model_builder.featurizers.DNASequenceFeaturizer'
          ? ['dna']
          : (option.classPath as StepValue['type']) ===
            'fleet.model_builder.featurizers.RNASequenceFeaturizer'
          ? ['rna']
          : (option.classPath as StepValue['type']) ===
            'fleet.model_builder.featurizers.ProteinSequenceFeaturizer'
          ? ['protein']
          : categoricalFeaturizerTypes.includes(
              option.classPath as StepValue['type']
            )
          ? ['categorical']
          : undefined,
    };

    if (
      option.classPath === 'fleet.model_builder.featurizers.MoleculeFeaturizer'
    )
      option.compatibleWith.framework = ['torch'];

    if (!option.compatibleWith?.domains) delete option.compatibleWith.domains;

    return option;
  };

  const preprocessingOptions = options.getPreprocessingOptions();

  const transformOptions = preprocessingOptions
    .filter((option) => option.type === 'transformer')
    .map(toConstructorArgsConfig) as StepValue[];

  const unconfiguredFeaturizerOptions = preprocessingOptions
    .filter((option) => option.type === 'featurizer')
    .map(applyCompatibilityAttributeMock);

  const filterColumnFeaturizersOptions = (column: SimpleColumnConfig) => {
    return unconfiguredFeaturizerOptions
      .filter((featurizer) => {
        let valid: boolean = true;

        if (!featurizer.compatibleWith) return true;

        if (featurizer.compatibleWith.domains) {
          valid =
            valid &&
            featurizer.compatibleWith.domains?.includes(
              column.dataType.domainKind
            );
        }
        if (featurizer.compatibleWith.framework) {
          valid =
            valid && featurizer.compatibleWith.framework?.includes('sklearn');
        }

        return valid;
      })
      .map(toConstructorArgsConfig) as StepValue[];
  };

  return (
    <>
      <Box sx={{ mb: 2, mt: 3 }}>
        <Text variant="subtitle1">Feature Columns:</Text>
        {featureColumns.map((column, index) => (
          <ColumnsPipelineInput
            key={index}
            column={{
              config: column,
              index,
              type: 'featureColumns',
            }}
            featurizerOptions={filterColumnFeaturizersOptions(column)}
            transformOptions={transformOptions}
          />
        ))}
      </Box>

      <Box>
        <Text variant="subtitle1">Target Columns:</Text>
        {targetColumns.map((column, index) => (
          <ColumnsPipelineInput
            key={index}
            column={{
              config: column,
              index,
              type: 'targetColumns',
            }}
            featurizerOptions={filterColumnFeaturizersOptions(column)}
            transformOptions={transformOptions}
          />
        ))}
      </Box>
    </>
  );
};

export default DataPreprocessingInput;
