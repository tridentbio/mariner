import { InputLabel, Switch, TextField, Box } from '@mui/material';
import CustomInputField from 'components/atoms/CustomInputField';
import useModelEditor from 'hooks/useModelEditor';
import { makeComponentEdit } from 'model-compiler/src/implementation/commands/EditComponentsCommand';
import { getComponent } from 'model-compiler/src/implementation/modelSchemaQuery';
import { LayerFeaturizerType } from 'model-compiler/src/interfaces/model-editor';
import EditorSelect from './EditorSelect';

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
  const { options, editComponent, schema, suggestionsByNode } =
    useModelEditor();
  if (!options) return null;
  const option = options[props.data.type!];
  if (!option || !option.component.constructorArgsSummary) return null;
  const editConstrutorArgs = (key: string, value: any) =>
    schema &&
    !editComponent({
      schema,
      data: makeComponentEdit({
        component: getComponent(schema, props.data.name),
        constructorArgs: {
          [key]: value,
        },
        options,
      }),
    });
  const suggestions = suggestionsByNode[props.data.name] || [];
  const errors = suggestions.reduce(
    (acc, sug) => ({ ...acc, ...sug.getConstructorArgsErrors() }),
    {} as Record<string, string>
  );
  return (
    <>
      <h1>hi</h1>
      <div style={{ marginTop: 5 }}>
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
                  editable={editable}
                  key={key}
                  option={option}
                  argKey={key}
                  editConstrutorArgs={editConstrutorArgs}
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
                  sx={{ mb: 2 }}
                  key={key}
                  onBlur={(event) =>
                    editConstrutorArgs(key, event.target.value)
                  }
                  error={key in errors}
                  label={errors[key] || key}
                  disabled={!editable}
                />
              );
            else if (isIntOrFloat(type)) {
              return (
                <CustomInputField
                  type="number"
                  inputMode="numeric"
                  key={key}
                  sx={{ mb: 2 }}
                  value={
                    props.data.constructorArgs &&
                    props.data.constructorArgs[
                      key as keyof typeof props.data.constructorArgs
                    ]
                  }
                  onBlur={(event) =>
                    editConstrutorArgs(key, parseFloat(event.target.value))
                  }
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
                    id={key}
                    className="nodrag"
                    defaultChecked={
                      props.data.constructorArgs &&
                      props.data.constructorArgs[
                        key as keyof typeof props.data.constructorArgs
                      ]
                    }
                    onChange={(event) =>
                      editConstrutorArgs(key, event.target.checked)
                    }
                    disabled={!editable}
                  />
                </Box>
              );
            else return null;
          })
          .filter((el) => !!el)}
      </div>
    </>
  );
};

export default ConstructorArgsInputs;
