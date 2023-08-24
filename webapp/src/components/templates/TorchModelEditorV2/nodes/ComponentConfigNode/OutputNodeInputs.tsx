import { InputLabel, Select, MenuItem } from '@mui/material';
import {
  AllowedLosses,
  useGetModelLossesQuery,
} from 'app/rtk/generated/models';
import useTorchModelEditor from 'hooks/useTorchModelEditor';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import { getComponent } from 'model-compiler/src/implementation/modelSchemaQuery';
import { Output } from '@model-compiler/src/interfaces/torch-model-editor';
import { useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';

export interface OutputNodeInputsProps {
  editable: boolean;
  name: string;
}

const notEditableMode = ({ editable }: { editable: boolean }) =>
  !editable &&
  `
  svg {
    display: none;
  }
  #mui-component-select-taskType {
    padding-right: 14px;
  }
  #mui-component-select-lossFunction {
    padding-right: 14px;
  }
`;

const Container = styled.div`
  margin-top: 5;
  ${notEditableMode}
`;

type Loss = {
  key: string;
  value: string;
};

const columnTypeMap = {
  binClass: 'binary',
  mcClass: 'multiclass',
  regr: 'regression',
};

const columnTypeMapInverse = Object.entries(columnTypeMap).reduce(
  (obj, [key, value]) => ({ ...obj, [value]: key }),
  {}
) as Record<'binary' | 'multiclass' | 'regression', keyof AllowedLosses>;

const friendlyColumnType = (columnType: keyof AllowedLosses): string =>
  (columnType &&
    columnType !== 'typeMap' &&
    columnTypeMap[columnType] &&
    columnTypeMap[columnType].replace(/./, (c) => c.toUpperCase())) ||
  '';

const getColumnType = (dataType: Output['dataType']): keyof AllowedLosses => {
  if (dataType.domainKind === 'categorical') {
    const numClasses = Object.keys(dataType.classes).length;
    if (numClasses === 2) return 'binClass';
    else return 'mcClass';
  } else return 'regr';
};

const OutputNodeInputs = ({ editable, name }: OutputNodeInputsProps) => {
  const { editComponent, schema } = useTorchModelEditor();

  const component = useMemo(
    () => schema && (getComponent(schema, name) as Output),
    [schema, name]
  );

  const [selectsOpened, setSelectsOpened] = useState([false, false]);

  const selectRefs = [
    useRef<HTMLSelectElement>(null),
    useRef<HTMLSelectElement>(null),
  ];

  const { data: allowedLosses } = useGetModelLossesQuery();

  const [taskType, setTaskType] = useState<keyof AllowedLosses | undefined>();
  useEffect(() => {
    if (editable)
      component && !taskType && setTaskType(getColumnType(component.dataType));
    else
      component &&
        component.columnType &&
        setTaskType(columnTypeMapInverse[component.columnType]);
  }, [component]);

  const losses = useMemo(() => {
    if (!allowedLosses || !taskType) return [] as Loss[];
    return allowedLosses[taskType] as Loss[];
  }, [allowedLosses, taskType]);

  const [selectedLoss, setSelectedLoss] = useState<Loss | undefined>();
  useEffect(() => {
    if (editable) losses.length && setSelectedLoss(losses[0]);
    else
      component &&
        component.lossFn &&
        setSelectedLoss(losses.find((l) => l.key === component.lossFn));
  }, [losses]);

  useEffect(() => {
    editable &&
      schema &&
      component &&
      editComponent({
        schema,
        data: makeComponentEdit({
          component,
          lossFn: selectedLoss?.key,
          columnType:
            (taskType && taskType !== 'typeMap' && columnTypeMap[taskType]) ||
            undefined,
        }),
      });
  }, [selectedLoss, taskType]);

  useEffect(() => {
    selectRefs.forEach((ref, i) => {
      ref &&
        ref.current &&
        editable &&
        ref.current.addEventListener('mousedown', () => {
          setSelectsOpened((prev) => {
            const newOpened = [...prev];
            newOpened[i] = !newOpened[i];
            return newOpened;
          });
        });
    });
  }, []);
  return (
    <Container editable={editable}>
      <InputLabel>Task Type</InputLabel>
      <Select
        sx={{
          mb: 2,
          width: '80%',
          maxHeight: 60,
          fontSize: 16,
          py: -5,
          color: 'grey.600',
          zIndex: 100,
        }}
        ref={selectRefs[0]}
        onChange={({ target }) => {
          setTaskType(target.value as keyof AllowedLosses);
        }}
        open={selectsOpened[0]}
        onClose={() => setSelectsOpened((prev) => [false, prev[1]])}
        value={taskType || ''}
        name={'taskType'}
      >
        {Object.keys(columnTypeMap).map((key) => (
          <MenuItem key={key} value={key}>
            {friendlyColumnType(key as keyof AllowedLosses)}
          </MenuItem>
        ))}
      </Select>

      <InputLabel>Loss Function</InputLabel>
      <Select
        sx={{
          mb: 2,
          width: '80%',
          maxHeight: 60,
          fontSize: 16,
          py: -5,
          color: 'grey.600',
          zIndex: 100,
        }}
        onChange={({ target }) => {
          const loss = losses.find((l) => l.key === target.value);
          setSelectedLoss(loss);
        }}
        open={selectsOpened[1]}
        onClose={() => setSelectsOpened((prev) => [prev[0], false])}
        ref={selectRefs[1]}
        value={selectedLoss?.value || ''}
        name={'lossFunction'}
      >
        {losses.map((loss: Loss) => (
          <MenuItem key={loss.key} value={loss.value}>
            {loss.value}
          </MenuItem>
        ))}
      </Select>
    </Container>
  );
};

export default OutputNodeInputs;
