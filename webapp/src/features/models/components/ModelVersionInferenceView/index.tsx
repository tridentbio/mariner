import {
  ElementRef,
  ReactNode,
  useCallback,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Box } from '@mui/system';
import { Text } from 'components/molecules/Text';
import {
  Model,
  ModelInputValue,
  ModelOutputValue,
} from 'app/types/domain/models';
import NoData from 'components/atoms/NoData';
import { Button } from '@mui/material';
import ModelPrediction from './ModelPrediction';
import ModelInput from './ModelInput';
import DataSummary from './DataSummary';
import { datasetsApi } from 'app/rtk/datasets';
import { getPrediction } from 'features/models/modelsApi';
import { TorchModelSpec } from '@app/rtk/generated/models';

interface ModelVersionInferenceViewProps {
  model: Model;
  modelVersionId: number;
}

interface SectionProps extends Record<string, any> {
  children: ReactNode;
  title: string;
}

const Section = ({ children, title, ...rest }: SectionProps) => {
  return (
    <Box sx={{ mb: 1 }}>
      <Text fontWeight="bold">{title}:</Text>
      <Box sx={{ ml: 1 }} {...rest}>
        {children}
      </Box>
    </Box>
  );
};

// TODO: update component to support all frameworks configs.
// Currently, only torch is supported, because it's dataset config is slightly different from 
// the usual, since it has a columnType property for each column.
const ModelVersionInferenceView = ({
  model,
  modelVersionId,
}: ModelVersionInferenceViewProps) => {
  const [modelInputs, setModelInputs] = useState<ModelInputValue>({});
  const [modelOutputs, setModelOutputs] = useState<ModelOutputValue>([]);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const { data: dataset } = datasetsApi.useGetDatasetByIdQuery(
    model.datasetId!
  );
  const inputRef = useRef<ElementRef<typeof ModelInput>>(null);
  const handlePrediction = () => {
    setPredictionLoading(true);
    getPrediction({ modelVersionId, modelInputs })
      .then(setModelOutputs)
      .finally(() => setPredictionLoading(false));
  };
  const resetModelInputs = () => {
    setModelInputs({});
  };
  const modelVersion = useMemo(() => {
    return model.versions.find(
      (modelVersion) => modelVersion.id === modelVersionId
    );
  }, [model, modelVersionId]);
  const targetColumns = useMemo(
    () => (modelVersion?.config as TorchModelSpec).dataset.targetColumns || [],
    [modelVersion?.config.dataset.targetColumns]
  );

  const isTargetColumnCategorical = useCallback(
    (columnName: string): boolean => {
      const targetColumn = targetColumns.find(
        (targetColumn) => targetColumn.name === columnName
      );
      return ['multiclass', 'binary'].includes(targetColumn?.columnType || '');
    },
    [targetColumns]
  );
  return (
    <Box>
      <Section title="Description">
        {modelVersion?.description ? (
          <Text>{modelVersion?.description}</Text>
        ) : (
          <NoData />
        )}
      </Section>
      {dataset && modelVersion && (
        <>
          <Section title="Inputs">
            <ModelInput
              ref={inputRef}
              value={modelInputs}
              onChange={(value) => setModelInputs(value)}
              config={modelVersion.config}
              columns={Object.keys(dataset.stats.full)}
              columnsMeta={dataset.columnsMetadata || []}
            />
            <Box sx={{ display: 'flex', flexDirection: 'row' }}>
              <Button
                sx={{ mr: 3 }}
                disabled={predictionLoading}
                onClick={handlePrediction}
                variant="contained"
                color="primary"
              >
                Predict
              </Button>
              <Button
                onClick={() => {
                  resetModelInputs();
                  inputRef.current?.reset();
                }}
                variant="contained"
                color="primary"
              >
                Reset
              </Button>
            </Box>
          </Section>
          <Section
            title="Predictions"
            sx={{
              display: 'flex',
              flexDirection: 'row',
              justifyContent: 'space-evenly',
            }}
          >
            {modelOutputs &&
              Object.keys(modelOutputs).map((key) => (
                <>
                  {isTargetColumnCategorical(key) ? (
                    <ModelPrediction
                      type={'categorical'}
                      // @ts-ignore
                      value={modelOutputs[key]}
                      column={key}
                    />
                  ) : (
                    <ModelPrediction
                      unit={(() => {
                        // TODO: check how to handle inference page with multiple target columns
                        const targetColumn =
                          targetColumns.find(
                            (targetColumn) => targetColumn.name === key
                          ) || targetColumns[0];
                        return 'unit' in targetColumn.dataType
                          ? targetColumn.dataType.unit
                          : '';
                      })()}
                      column={key}
                      type="numerical"
                      // @ts-ignore
                      value={modelOutputs[key]}
                    />
                  )}
                </>
              ))}
          </Section>

          <Section title="Data Summary">
            <DataSummary
              columnsData={dataset.stats}
              // TODO: check what is the deal of this props on DataSummary
              inferenceValue={modelOutputs ? 0 : undefined}
              inferenceColumn={targetColumns[0].name}
            />
          </Section>
        </>
      )}
    </Box>
  );
};

export default ModelVersionInferenceView;
