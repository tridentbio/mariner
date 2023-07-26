import { Box, Divider, SxProps } from '@mui/material';
import {
  Experiment,
  Model,
  useGetExperimentsMetricsForModelVersionQuery,
} from 'app/rtk/generated/models';
import Select from 'components/molecules/CenteredSelect';
import { MetricsAccordionPlot } from 'components/templates/MetricsAccordionPlot';
import { useState } from 'react';

type ModelMetricsViewProps = {
  model: Model;
};

const containerSx: SxProps = {
  display: 'flex',
  flexDirection: 'row',
  gap: '2rem',
  justifyContent: 'space-between',
};

const ModelMetricsView = ({ model }: ModelMetricsViewProps) => {
  const [selectedModelVersionId, setSelectedModelVersionId] =
    useState<number>(-1);

  const [selectedExperiment, setSelectedExperiment] =
    useState<Experiment | null>(null);

  const { data: currentExperiments = [] } =
    useGetExperimentsMetricsForModelVersionQuery({
      modelVersionId: selectedModelVersionId,
    });

  return (
    <>
      <Box sx={containerSx}>
        <Select
          title="Model Version"
          items={model.versions}
          keys={{ value: 'id', children: 'name' }}
          value={selectedModelVersionId}
          onChange={({ target }) =>
            setSelectedModelVersionId(target.value as number)
          }
        />

        {selectedModelVersionId !== -1 ? (
          <Select
            title="Experiment"
            items={currentExperiments}
            keys={{ value: 'experimentName', children: 'experimentName' }}
            value={selectedExperiment?.experimentName || ''}
            onChange={({ target }) => {
              setSelectedExperiment(
                currentExperiments.find(
                  (experiment) => experiment.experimentName === target.value
                ) || null
              );
            }}
          />
        ) : (
          <Box sx={{ width: '100%' }} />
        )}
      </Box>
      <Divider sx={{ my: '2rem' }} />
      {selectedExperiment && selectedExperiment.history && (
        <MetricsAccordionPlot experiment={selectedExperiment} />
      )}
    </>
  );
};

export default ModelMetricsView;
