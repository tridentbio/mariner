import { useAppSelector } from 'app/hooks';
import {
  Dataset,
  useLazyGetMyDatasetQuery,
  useLazyGetMyDatasetsQuery,
} from 'app/rtk/generated/datasets';
import { ModelCreate } from 'app/rtk/generated/models';
import { DataTypeGuard } from 'app/types/domain/datasets';
import DatasetSelect from 'features/datasets/components/DatasetSelect';
import ColumnDescriptionSelector from 'features/models/components/ColumnDescriptionSelector';
import { useEffect, useMemo, useState } from 'react';
import { Control, Controller, useFormContext, useWatch } from 'react-hook-form';
import { useSearchParams } from 'react-router-dom';

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
  const { setValue, watch, getValues } = useFormContext();
  const [fetchDatasetById] = useLazyGetMyDatasetQuery();
  const [fetchDatasets] = useLazyGetMyDatasetsQuery();
  const [searchParams] = useSearchParams();

  const datasetName = useWatch({
    control,
    name: 'config.dataset.name',
  });

  const targetColumns = useWatch({
    control,
    name: 'config.dataset.targetColumns',
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
  const selectOptions = useMemo(() => {
    return (dataset?.columnsMetadata || []).map((col) => ({
      name: col.pattern,
      dataType: col.dataType,
    }));
  }, [dataset]);

  const targetSelectOptions = useMemo(() => {
    return selectOptions.filter(
      (col) =>
        DataTypeGuard.isCategorical(col.dataType) ||
        DataTypeGuard.isNumericalOrQuantity(col.dataType)
    );
  }, [selectOptions]);

  const featureSelectOptions = useMemo(() => {
    if (!targetColumns.length) return selectOptions;

    return selectOptions.filter((col) => {
      const declaredInTargetColumnsList = targetColumns.some(
        (targetCol) => targetCol.name == col.name
      );

      return !declaredInTargetColumnsList;
    });
  }, [targetColumns, selectOptions]);

  const removeTargetColumnsFromFeatureColumnsList = () => {
    const featureColumnsFieldValue: typeof featureSelectOptions = getValues(
      'config.dataset.featureColumns'
    );

    const columnToRemove = featureColumnsFieldValue.find((featureCol) => {
      return !featureSelectOptions.some((col) => col.name == featureCol.name);
    });

    if (columnToRemove) {
      const updatedFeatureColumnsFieldValue = featureColumnsFieldValue.filter(
        (featureCol) => featureCol.name != columnToRemove.name
      );

      setValue(
        'config.dataset.featureColumns',
        updatedFeatureColumnsFieldValue
      );
    }
  };

  useEffect(removeTargetColumnsFromFeatureColumnsList, [targetColumns]);

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
        render={({ field, fieldState }) => {
          if (!dataset) return <div />;
          return (
            <ColumnDescriptionSelector
              id="target-col"
              data-testid="dataset-target-column"
              multiple
              onBlur={field.onBlur}
              error={!!fieldState.error?.ref}
              value={field.value}
              onChange={(colDescription) => {
                field.onChange({
                  target: {
                    value: colDescription,
                  },
                });
              }}
              label={fieldState.error?.message || 'Target Column'}
              options={targetSelectOptions}
            />
          );
        }}
      />

      <Controller
        control={control}
        name="config.dataset.featureColumns"
        render={({ field, fieldState }) =>
          !dataset ? (
            <div />
          ) : (
            <ColumnDescriptionSelector
              onBlur={field.onBlur}
              error={!!fieldState.error?.ref}
              id="feature-cols"
              data-testid="dataset-feature-columns"
              multiple
              value={field.value}
              onChange={(featureCols) =>
                field.onChange({ target: { value: featureCols } })
              }
              label={fieldState.error?.message || 'Feature Column'}
              options={featureSelectOptions}
            />
          )
        }
      ></Controller>
    </div>
  );
};

export default FeatureAndTargets;
