import { TargetConfig } from '@app/rtk/generated/models';
import { ModelOutputValue } from '@app/types/domain/models';
import ModelPrediction from '@features/models/components/ModelVersionInferenceView/ModelPrediction';

export const InferenceOutput = ({
  outputValues,
  targetColumns,
}: {
  outputValues: ModelOutputValue;
  targetColumns: TargetConfig[];
}) => (
  <>
    {Object.keys(outputValues).map((key) => {
      const column = targetColumns.find((column) => column.name === key);
      if (!column) return null;

      const type =
        column.columnType == 'regression' ? 'numerical' : 'categorical';

      return (
        <ModelPrediction
          key={column.name}
          column={column.name}
          unit={'unit' in column.dataType ? column.dataType.unit : ''}
          type={type}
          // @ts-ignore
          value={outputValues[column.name]}
        />
      );
    })}
  </>
);
