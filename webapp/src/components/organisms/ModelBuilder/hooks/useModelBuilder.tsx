import { ReactNode, createContext, useContext, useState } from 'react';

interface ModelBuilderContextProps {
  editable: boolean;
  setEditable: React.Dispatch<React.SetStateAction<boolean>>;
  defaultExpanded: boolean;
}

// @ts-ignore
const ModelBuilderContext = createContext<ModelBuilderContextProps>({});

export const ModelBuilderContextProvider = ({
  children,
  editable = true,
  defaultExpanded = false,
}: {
  children?: ReactNode;
  editable?: boolean;
  defaultExpanded?: boolean;
}) => {
  const [isEditable, setIsEditable] = useState<boolean>(editable);

  return (
    <ModelBuilderContext.Provider
      value={{
        editable: isEditable,
        setEditable: setIsEditable,
        defaultExpanded,
      }}
    >
      {children}
    </ModelBuilderContext.Provider>
  );
};

const useModelBuilder = () => {
  const value = useContext(ModelBuilderContext);

  return { ...value };
};

export default useModelBuilder;
