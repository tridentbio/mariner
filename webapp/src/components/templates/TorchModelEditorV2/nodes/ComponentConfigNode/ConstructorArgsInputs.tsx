import { InputLabel, Switch, TextField, Box } from '@mui/material';
import CustomInputField from 'components/atoms/CustomInputField';
import useTorchModelEditor from 'hooks/useTorchModelEditor';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import { getComponent } from 'model-compiler/src/implementation/modelSchemaQuery';
import { LayerFeaturizerType } from '@model-compiler/src/interfaces/torch-model-editor';
import EditorSelect from './EditorSelect';
import { useEffect, useMemo, useState } from 'react';

export interface ConstructorArgsInputsProps {
  data: LayerFeaturizerType;
  editable: boolean;
}

const isString = (type: any) =>
  typeof type === 'string' && type.includes('str');

const isIntOrFloat = (type: any) =>
  typeof type === 'string' && (type.includes('int') || type.includes('float'));

const isBoolean = (type: any) =>
  typeof type === 'string' && type.includes('bool');

const ConstructorArgsInputs = ({
  editable,
  ...props
}: ConstructorArgsInputsProps) => {
  const { options, editComponent, schema, suggestionsByNode, setNodes } =
    useTorchModelEditor();

  const [argsForm, setArgsForm] = useState<{ [field: string]: any }>({});

  const editConstrutorArgs = () => {
    if (schema && editable) {
      editComponent(
        {
          data: makeComponentEdit({
            component: getComponent(schema, props.data.name),
            constructorArgs: argsForm,
            options,
          }),
        },
        schema
      );

      //? Persist selected node overlay
      setNodes((prev) =>
        prev.map((node) => ({
          ...node,
          selected: node.id === props.data.name ? true : node.selected,
        }))
      );
    }
  };

  useEffect(() => {
    editConstrutorArgs();
  }, [argsForm]);

  const suggestions = suggestionsByNode[props.data.name] || [];
  const errors = suggestions.reduce(
    (acc, sug) => ({ ...acc, ...sug.getConstructorArgsErrors() }),
    {} as Record<string, string>
  );

  const option = options?.[props.data.type!];

  const ArgsList = useMemo(() => {
    if (!option) return null;

    return (
      <>
        {Object.entries(option.component.constructorArgsSummary)
          .map(([key, type]) => {
            if (
              !('constructorArgs' in props.data) ||
              (props.data.constructorArgs &&
                !(key in props.data.constructorArgs))
            ) {
              return null;
            } else if (
              isString(type) &&
              !!option.argsOptions &&
              option.argsOptions[key]
            ) {
              return (
                <EditorSelect
                  data-testid={`${props.data.name}-${key}`}
                  editable={editable}
                  key={key}
                  option={option}
                  argKey={key}
                  onChange={(value) =>
                    setArgsForm((prev) => ({ ...prev, [key]: value }))
                  }
                  errors={errors}
                  value={
                    (props.data.constructorArgs &&
                      props.data.constructorArgs[
                        key as keyof typeof props.data.constructorArgs
                      ]) ||
                    ''
                  }
                ></EditorSelect>
              );
            } else if (isString(type))
              return (
                <TextField
                  data-testid={`${props.data.name}-${key}`}
                  sx={{ mb: 2 }}
                  key={key}
                  onBlur={(event) => {
                    setArgsForm((prev) => ({
                      ...prev,
                      [key]: event.target.value,
                    }));
                  }}
                  error={key in errors}
                  label={errors[key] || key}
                  disabled={!editable}
                />
              );
            else if (isIntOrFloat(type)) {
              return (
                <CustomInputField
                  data-testid={`${props.data.name}-${key}`}
                  type="number"
                  inputMode="numeric"
                  key={key}
                  sx={{ mb: 2 }}
                  value={
                    argsForm[key] ||
                    props.data.constructorArgs?.[
                      key as keyof typeof props.data.constructorArgs
                    ]
                  }
                  onBlur={(event) => {
                    setArgsForm((prev) => ({
                      ...prev,
                      [key]: parseFloat(event.target.value),
                    }));
                  }}
                  error={key in errors}
                  label={errors[key] || key}
                  disabled={!editable}
                  inputProps={{
                    step: '0.0001',
                  }}
                />
              );
            } else if (isBoolean(type))
              return (
                <Box sx={{ marginBottom: 1 }} key={key}>
                  <InputLabel error={key in errors} id={`label-${key}`}>
                    {errors[key] || key}
                  </InputLabel>
                  <Switch
                    data-testid={`${props.data.name}-${key}`}
                    id={key}
                    className="nodrag"
                    defaultChecked={
                      props.data.constructorArgs &&
                      (props.data.constructorArgs[
                        key as keyof typeof props.data.constructorArgs
                      ] ||
                        null)
                    }
                    onChange={(event) => {
                      setArgsForm((prev) => ({
                        ...prev,
                        [key]: event.target.checked,
                      }));
                    }}
                    disabled={!editable}
                  />
                </Box>
              );
            else return null;
          })
          .filter((el) => !!el)}
      </>
    );
  }, [editable, props.data, option]);

  if (!options) return null;
  if (!option || !option.component.constructorArgsSummary) return null;

  return ArgsList;
};

export default ConstructorArgsInputs;
