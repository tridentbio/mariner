import { Box, Divider, SxProps } from '@mui/material';
import {
  Experiment,
  Model,
  useGetExperimentsMetricsForModelVersionQuery,
} from 'app/rtk/generated/models';
import Select from 'components/molecules/CenteredSelect';
import { MetricsAccordionPlot } from 'components/templates/MetricsAccordionPlot';
import { useState } from 'react';
import ModelVersionSelect from './ModelVersionSelect';

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

  const onModelVersionChange = (modelVersionId: number) => {
    setSelectedModelVersionId(modelVersionId);
    setSelectedExperiment(null);
  };

  const FrameworkExperimentMetrics = ({
    experiment,
  }: {
    experiment: Experiment;
  }) => {
    const experimentModelVersion = model.versions.find(
      (version) => version.id === experiment.modelVersionId
    );

    if (!experimentModelVersion) return null;

    switch (experimentModelVersion.config.framework) {
      case 'torch':
        return <MetricsAccordionPlot experiment={experiment} />;
      default:
        return null;
    }
  };

  return (
    <>
      <Box sx={containerSx}>
        <ModelVersionSelect
          sx={{ width: '100%', display: 'flex', alignItems: 'end' }}
          disableClearable
          model={model}
          value={model.versions.find(
            (version) => version.id === selectedModelVersionId
          )}
          onChange={(modelVersion) =>
            modelVersion && onModelVersionChange(modelVersion.id)
          }
        />

        <Select
          sx={{ width: '100%' }}
          title="Experiment"
          disabled={!currentExperiments.length}
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
      </Box>
      <Divider sx={{ my: '2rem' }} />
      {selectedExperiment && selectedExperiment.history && (
        <FrameworkExperimentMetrics experiment={selectedExperiment} />
      )}
    </>
  );
};

export default ModelMetricsView;
