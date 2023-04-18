import { Accordion, AccordionDetails, AccordionSummary } from '@mui/material';
import { Experiment } from 'app/rtk/generated/models';
import { useMemo } from 'react';
import styled from 'styled-components';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { MetricPlot } from 'components/organisms/MetricPlot';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2rem;
  justify-content: space-evenly;
  max-width: 100%;

  .MuiPaper-root {
    width: 100%;
  }

  .MuiAccordionSummary-content {
    font-size: 25px;
  }

  .MuiAccordionDetails-root {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 2rem;
    justify-content: space-evenly;
  }
`;

const StageAccordion = ({
  children,
  stage,
}: {
  children: React.ReactNode;
  stage: string;
}) => {
  return (
    <Accordion>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        {stage} Stage.
      </AccordionSummary>
      <AccordionDetails>{children}</AccordionDetails>
    </Accordion>
  );
};

type MetricsPlotProps = {
  experiment: Experiment;
};

export const MetricsAccordionPlot = ({ experiment }: MetricsPlotProps) => {
  if (!experiment.history) return null;

  const epochs = useMemo(() => {
    return (
      experiment.history && Array.from(new Set(experiment.history['epoch']))
    );
  }, [experiment.history]);

  const metricsByStage = useMemo(() => {
    if (!experiment.history) return {};

    const separatedMetrics = {
      Training: {},
      Validation: {},
    } as {
      [key: string]: {
        [key: string]: number[];
      };
    };

    Object.keys(experiment.history).forEach((key) => {
      if (key === 'epoch') return;
      else if (key.startsWith('train'))
        separatedMetrics.Training[key] = experiment.history![key];
      else if (key.startsWith('val'))
        separatedMetrics.Validation[key] = experiment.history![key];
    });

    return separatedMetrics;
  }, [experiment.history]);

  return (
    <Container>
      {Object.keys(metricsByStage).map((stage) => (
        <StageAccordion key={stage} stage={stage}>
          {Object.keys(metricsByStage[stage]).map((key) => {
            const metrics = useMemo<number[]>(
              () => experiment.history![key] || [],
              [experiment.history]
            );

            if (!epochs || epochs.length !== metrics.length) return null;

            return (
              <MetricPlot
                key={key}
                epochs={epochs}
                metricValues={metrics}
                metricName={key}
              />
            );
          })}
        </StageAccordion>
      ))}
    </Container>
  );
};
