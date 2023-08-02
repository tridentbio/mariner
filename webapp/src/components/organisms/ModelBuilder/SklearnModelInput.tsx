import { SklearnModelSpec } from "@app/rtk/generated/models";
import useModelOptions, { toConstructorArgsConfig } from "@hooks/useModelOptions";
import { useMemo } from "react";
import PreprocessingStepSelect from "./PreprocessingStepSelect";

type SklearnModelConfig = SklearnModelSpec['spec']
export interface SklearnModelInputProps {
  value?: SklearnModelConfig;
  onChange?: (value: SklearnModelConfig | null) => void
}

export default function SklearnModelInput(props: SklearnModelInputProps) {
  const { value, onChange } = props
  const {getScikitOptions} = useModelOptions()
  const options = useMemo(() => {
    return getScikitOptions().map(toConstructorArgsConfig)
  }, [getScikitOptions])
  return (
    <div>
      <PreprocessingStepSelect
        label={"Sklearn Model"}
        options={options}
        value={undefined}
        onChange={value => console.log(value)}
      />
    </div>
  );
}

