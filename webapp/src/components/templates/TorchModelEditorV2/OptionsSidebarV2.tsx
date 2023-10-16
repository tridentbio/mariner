import { Box } from '@mui/system';
import * as modelsApi from 'app/rtk/generated/models';
import { useGetModelOptionsQuery } from 'app/rtk/generated/models';
import { Text } from 'components/molecules/Text';
import DocsModel from 'components/templates/TorchModelEditor/Components/DocsModel/DocsModel';
import { DragEvent, useMemo } from 'react';
import { substrAfterLast } from 'utils';

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
}

/**
 * The sidebar of the model editor that shows layers and featurizers options.
 */
const OptionsSidebarV2 = ({ onDragStart }: OptionsSidebarProps) => {
  const { data } = useGetModelOptionsQuery();
  // Hack to hide some featurizers
  const modelOptions = data || [];

  const IGNORED_LIBS = ['sklearn'];

  const modelsByLib = useMemo(() => {
    return (
      modelOptions?.reduce((acc, model) => {
        const lib = model.classPath.split('.')[0];

        if (IGNORED_LIBS.includes(lib)) return acc;

        if (!acc[lib]) {
          acc[lib] = [];
        }
        acc[lib].push(model);
        return acc;
      }, {} as Record<string, typeof modelOptions>) || {}
    );
  }, [modelOptions]);

  return (
    <>
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
    </>
  );
};

export default OptionsSidebarV2;
