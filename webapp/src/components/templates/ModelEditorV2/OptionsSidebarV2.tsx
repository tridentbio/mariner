import { useState, useMemo, DragEvent } from 'react';
import { RemoveSharp, TurnLeft } from '@mui/icons-material';
import { IconButton } from '@mui/material';
import { Box } from '@mui/system';
import { Text } from 'components/molecules/Text';
import { ArrayElement, substrAfterLast } from 'utils';
import { useGetModelOptionsQuery } from 'app/rtk/generated/models';
import DocsModel from 'components/templates/ModelEditor/Components/DocsModel/DocsModel';
import * as modelsApi from 'app/rtk/generated/models';
import { ModelSchema } from 'model-compiler/src/interfaces/model-editor';

export type HandleProtoDragStartParams = {
  event: DragEvent<HTMLDivElement>;
  data: modelsApi.ComponentOption;
};

const styleProto = {
  cursor: 'grab',
  p: 1,
  width: 'fit-content',
  border: '1px solid black',
  m: 1,
  borderRadius: 2,
};

interface OptionsSidebarProps {
  onDragStart: (params: HandleProtoDragStartParams) => void;
  editable?: boolean;
}

/**
 * The sidebar of the model editor that shows layers and featurizers options.
 */
const OptionsSidebarV2 = ({
  onDragStart,
  editable = true,
}: OptionsSidebarProps) => {
  const { data } = useGetModelOptionsQuery();
  // Hack to hide some featurizers
  const modelOptions = data || [];
  const [isModelOptionsOpened, setIsModelOptionsOpened] = useState(false);
  const handleToggleNodesRetraction = () => {
    setIsModelOptionsOpened((value) => !value);
  };

  const modelsByLib = useMemo(() => {
    return (
      modelOptions?.reduce((acc, model) => {
        const lib = model.classPath.split('.')[0];
        if (!acc[lib]) {
          acc[lib] = [];
        }
        acc[lib].push(model);
        return acc;
      }, {} as Record<string, typeof modelOptions>) || {}
    );
  }, [modelOptions]);

  const sidebarStyleRight = isModelOptionsOpened ? '0' : '-450px';

  const top = 0;
  return (
    <Box
      sx={{
        ...(editable ? {} : { display: 'none' }),
        position: 'absolute',
        top,
        p: 2,
        width: 400,
        right: sidebarStyleRight,
        zIndex: 200,
        borderRadius: 2,
        boxShadow: 'rgba(0,0,0,0.24) 0px 3px 8px',
        transition: 'right 0.6s',
        backgroundColor: 'white',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', flexDirection: 'row' }}>
        <IconButton color="primary" onClick={handleToggleNodesRetraction}>
          <RemoveSharp />
        </IconButton>
        {!isModelOptionsOpened && (
          <IconButton
            data-testid="openOptionsSidebarButton"
            color="primary"
            sx={{
              zIndex: 200,
              position: 'absolute',
              right: 450,
              // border: '3px solid red',
              top: top + 100,
            }}
            onClick={() => handleToggleNodesRetraction()}
          >
            <TurnLeft />
          </IconButton>
        )}
        <Text>You can drag these nodes to the editor</Text>
      </Box>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {Object.entries(modelsByLib).map(([lib, models], _index) => (
          <Box key={lib}>
            <Text ml={2} fontSize="1.2rem" fontWeight="600" marginTop="0.5rem">
              {lib === 'model_builder'
                ? 'mariner'.toUpperCase()
                : lib.toUpperCase()}
            </Text>
            <Box>
              <div
                style={{
                  flexDirection: 'row',
                  display: 'flex',
                  flexWrap: 'wrap',
                }}
              >
                {models.map((option) => (
                  <Box
                    sx={styleProto}
                    draggable
                    onDragStart={(event) => {
                      onDragStart({
                        event,
                        data: option,
                      });
                    }}
                    key={option.classPath}
                  >
                    {substrAfterLast(option.classPath, '.')}
                    <DocsModel
                      commonIconProps={{
                        fontSize: 'small',
                      }}
                      docs={option.docs}
                      docsLink={option.docsLink || ''}
                    />
                  </Box>
                ))}
              </div>
            </Box>
          </Box>
        ))}
      </div>
    </Box>
  );
};

export default OptionsSidebarV2;
