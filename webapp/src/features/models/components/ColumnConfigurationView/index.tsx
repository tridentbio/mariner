import { DatasetConfig } from '@app/rtk/generated/models';
import { ArrayElement } from '@utils';
import ColumnConfigurationAccordion from './ColumnConfigAccordion';
import { FormColumns } from './types';
import ColumnPreprocessingPipelineInput from './ColumnPreprocessingPipelineForm';
import { PreprocessingConfig } from './types';

const ColumnConfigurationView = ({
  formColumns,
  datasetConfig,
  setValue,
}: {
  formColumns: FormColumns;
  datasetConfig: DatasetConfig;
  setValue: (k: string, v: any) => void;
}) => {
  const addTransformerFunction =
    (col: ArrayElement<FormColumns['feature']>) =>
    (
      component: PreprocessingConfig,
      transformerGroup: 'transforms' | 'featurizers' = 'transforms'
    ) => {
      const transformers = [col.col, ...col.featurizers, ...col.transforms];
      const lastTransform = transformers.at(-1)!;

      const fowardArgs = Object.fromEntries(
        Object.entries(component.fowardArgs).map(([key, _]) => [
          key,
          `$${lastTransform.name}`,
        ])
      );
      const newTransformer = {
        ...component,
        name: `${transformerGroup.replace(/s$/g, '')}-${component.name}-${
          transformers.length
        }`,
        fowardArgs,
      };

      setValue(`config.dataset.${transformerGroup}`, [
        ...(datasetConfig[transformerGroup] || []),
        newTransformer,
      ]);
    };

  const deleteTransformerFunction =
    (col: ArrayElement<FormColumns['feature']>) =>
    (
      component: PreprocessingConfig,
      transformerGroup: 'transforms' | 'featurizers' = 'transforms'
    ) => {
      const transformers = [col.col, ...col.featurizers, ...col.transforms];
      const index = transformers.findIndex((t) => t.name === component.name);

      if (index === -1) return;

      transformers.splice(index, 1);

      setValue(`config.dataset.${transformerGroup}`, transformers);
    };

  return (
    <>
      {formColumns.feature.map((formColumn) => (
        <ColumnConfigurationAccordion
          key={formColumn.col.name}
          name={formColumn.col.name}
          dataType={formColumn.col.dataType}
        >
          <ColumnPreprocessingPipelineInput
            formColumn={formColumn}
            addTransformer={addTransformerFunction(formColumn)}
          />
        </ColumnConfigurationAccordion>
      ))}

      {formColumns.target.map((formColumn) => (
        <ColumnConfigurationAccordion
          key={formColumn.col.name}
          name={formColumn.col.name}
          dataType={formColumn.col.dataType}
          textProps={{ fontWeight: 'bold' }}
        >
          <ColumnPreprocessingPipelineInput
            formColumn={formColumn}
            addTransformer={addTransformerFunction(formColumn)}
          />
        </ColumnConfigurationAccordion>
      ))}
    </>
  );
};

export default ColumnConfigurationView;
