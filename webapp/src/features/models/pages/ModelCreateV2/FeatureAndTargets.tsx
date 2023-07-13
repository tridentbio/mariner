import { useEffect, useMemo, useState } from 'react';
import {
  Dataset,
  useLazyGetMyDatasetQuery,
  useLazyGetMyDatasetsQuery,
} from 'app/rtk/generated/datasets';
import DatasetSelect from 'features/datasets/components/DatasetSelect';
import ColumnDescriptionSelector from 'features/models/components/ColumnDescriptionSelector';
import { ModelCreate } from 'app/rtk/generated/models';
import { Control, Controller, useFormContext, useWatch } from 'react-hook-form';
import { useSearchParams } from 'react-router-dom';
import { DataTypeGuard } from 'app/types/domain/datasets';
import { useAppSelector } from 'app/hooks';
import { MenuItem, Select } from '@mui/material';

export interface FeatureAndTargets {
  control: Control<ModelCreate>;
}

const sortColumnsMetadata = (columnsMetadata: Dataset['columnsMetadata']) => {
  if (!columnsMetadata) return columnsMetadata;
  const arr = [...columnsMetadata];
  arr?.sort((a, b) => b.pattern.length - a.pattern.length);
  arr?.sort((a, b) =>
    (b.dataType.domainKind || '').localeCompare(a.dataType.domainKind || '')
  );
  return arr;
};
const FeatureAndTargets = ({ control }: FeatureAndTargets) => {
  const { setValue } = useFormContext();
  const [fetchDatasetById] = useLazyGetMyDatasetQuery();
  const [fetchDatasets] = useLazyGetMyDatasetsQuery();
  const [searchParams] = useSearchParams();

  const datasetName = useWatch({
    control,
    name: 'config.dataset.name',
  });

  const datasetInStore = useAppSelector((store) =>
    store.datasets.datasets.find((ds) => ds.name === datasetName)
  );
  useEffect(() => {
    if (!datasetInStore && datasetName) {
      fetchDatasets({
        page: 0,
        perPage: 15,
        searchByName: datasetName,
      });
    }
  }, []);
  // @ts-ignore
  const [dataset, setDataset] = useState<Dataset | undefined>(datasetInStore);
  const targetFeatureSelectOptions = useMemo(() => {
    return (dataset?.columnsMetadata || []).map((col) => ({
      name: col.pattern,
      dataType: col.dataType,
    }));
  }, [dataset]);

  const datasetId = searchParams.get('datasetId');
  useEffect(() => {
    if (!datasetId) return;
    const go = async () => {
      if (datasetId) {
        const datasetIdInt = parseInt(datasetId);
        if (isNaN(datasetIdInt)) return;
        const result = await fetchDatasetById({
          datasetId: parseInt(datasetId),
        });
        if ('data' in result && result.data) {
          // set form config.dataset.name
          setDataset(result.data);
          setValue('config.dataset.name', result.data.name);
        }
      }
    };
    go();
  }, [datasetId]);

  return (
    <div data-testid="dataset-config-form">
      <Controller
        control={control}
        name="config.dataset.name"
        rules={{
          required: { value: true, message: 'Dataset is required' },
        }}
        render={({ field, fieldState }) => (
          <DatasetSelect
            data-testid="dataset-selector"
            value={dataset}
            onBlur={field.onBlur}
            error={!!fieldState.error}
            label={fieldState.error?.message || 'Dataset'}
            onChange={(ds) => {
              if (ds) {
                field.onChange({ target: { value: ds.name } });
                setDataset(ds);
              } else {
                field.onChange({ target: { value: undefined } });
                setDataset(undefined);
              }
            }}
          />
        )}
      />
      <Controller
        control={control}
        name="config.dataset.targetColumns"
        rules={{
          required: { value: true, message: 'A target column is required' },
        }}
        render={({ field, fieldState }) => {
          if (!dataset) return <div />;
          return (
            <ColumnDescriptionSelector
              id="target-col"
              data-testid="dataset-target-column"
              multiple
              onBlur={field.onBlur}
              error={!!fieldState.error}
              value={field.value}
              onChange={(colDescription) => {
                field.onChange({
                  target: {
                    value: colDescription,
                  },
                });
              }}
              label={fieldState.error?.message || 'Target Column'}
              options={targetFeatureSelectOptions.filter(
                (col) =>
                  DataTypeGuard.isCategorical(col.dataType) ||
                  DataTypeGuard.isNumerical(col.dataType) ||
                  DataTypeGuard.isQuantity(col.dataType)
              )}
            />
          );
        }}
      />

      <Controller
        control={control}
        name="config.dataset.featureColumns"
        rules={{
          required: {
            value: true,
            message: 'The feature columns is required',
          },
          minLength: {
            value: 1,
            message: 'The feature columns must not be empty',
          },
        }}
        render={({ field, fieldState }) =>
          !dataset ? (
            <div />
          ) : (
            <ColumnDescriptionSelector
              onBlur={field.onBlur}
              error={!!fieldState.error}
              id="feature-cols"
              data-testid="dataset-feature-columns"
              multiple
              value={field.value}
              onChange={(featureCols) =>
                field.onChange({ target: { value: featureCols } })
              }
              label={fieldState.error?.message || 'Feature Column'}
              options={targetFeatureSelectOptions}
            />
          )
        }
      ></Controller>
    </div>
  );
};

export default FeatureAndTargets;
