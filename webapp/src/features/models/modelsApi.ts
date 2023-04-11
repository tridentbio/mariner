import { ModelInputValue, ModelOutputValue } from 'app/types/domain/models';
import api from '../../app/api';

export const getPrediction = async (predictionRequest: {
  modelVersionId: number;
  modelInputs: ModelInputValue;
}): Promise<ModelOutputValue> => {
  return api
    .post(
      `api/v1/models/${predictionRequest.modelVersionId}/predict`,
      predictionRequest.modelInputs
    )
    .then((res) => res.data);
};
