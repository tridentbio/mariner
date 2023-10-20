import { ModelOutputValue } from 'app/types/domain/models';
import { Text } from 'components/molecules/Text';
import { modelOutputToVegaSpec } from 'components/molecules/Vega/VerticalHistogram';
import { extractVal } from 'features/models/common';
import { Vega } from 'react-vega';
import styled from 'styled-components';

type CategoricalPredictionProps = {
  type: 'categorical';
  column: string;
  value: ModelOutputValue;
};

type NumericalPredictionProps = {
  type: 'numerical';
  column: string;
  value: ModelOutputValue;
  unit: string;
};
type ModelPredictionProps =
  | CategoricalPredictionProps
  | NumericalPredictionProps;

const NumericalPredictionContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 1rem;
`;

/**
 * Component responsible for rendering the model
 * prediction.
 * If the model has a categorical output, it is rendered
 * as a histogram
 * If the model has a numerical output, it is rendered as a
 * constant value on a histogram of the dataset's target
 * distribution.
 * If the model has a multicategorical output, ... TODO
 */
const ModelPrediction = ({
  value,
  type,
  column,
  ...props
}: ModelPredictionProps) => {
  return (
    <div data-testid="inference-result">
      <Text fontWeight="bold">{`Prediction for ${column}:`}</Text>

      {type === 'categorical' && (
        // @ts-ignore
        <Vega spec={modelOutputToVegaSpec(value['probs'], value['classes'])} />
      )}

      {type === 'numerical' && 'unit' in props && (
        <NumericalPredictionContainer>
          <Text fontWeight="bold">
            {extractVal(value)!.toExponential(2)} {props.unit}
          </Text>
        </NumericalPredictionContainer>
      )}
    </div>
  );
};

export default ModelPrediction;
