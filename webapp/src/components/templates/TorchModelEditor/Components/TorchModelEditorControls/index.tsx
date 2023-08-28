import { ArrowDownward, ArrowForward } from '@mui/icons-material';
import React, { useState } from 'react';
import { ControlButton, Controls } from 'reactflow';
import CustomAutoLayoutButton from '../CustomAutoLayoutButton';
import OpenInFullIcon from '@mui/icons-material/OpenInFull';
import CloseFullscreenIcon from '@mui/icons-material/CloseFullscreen';
import useTorchModelEditor from 'hooks/useTorchModelEditor';
import { Tooltip } from '@mui/material';

type TorchModelEditorControlsProps = {
  spacementMultiplierState: [
    number,
    React.Dispatch<React.SetStateAction<number>>
  ];
  autoLayout: {
    horizontal: (multiplier: number) => void;
    vertical: (multiplier: number) => void;
  };
  contentEditable?: boolean;
};

const TorchModelEditorControls: React.FC<TorchModelEditorControlsProps> = ({
  spacementMultiplierState,
  autoLayout,
  contentEditable = true,
}) => {
  const { expandNodes, contractNodes } = useTorchModelEditor();
  const [allNodesExpanded, setAllNodesExpanded] = useState(true);
  const [slidersState, setSlidersState] = useState({
    horizontal: false,
    vertical: false,
  });
  const [spacementMultiplier, setSpacementMultiplier] =
    spacementMultiplierState;
  return (
    <Controls showInteractive={contentEditable}>
      <CustomAutoLayoutButton
        onAction={(multiplier) => {
          autoLayout.vertical(multiplier);
          setSpacementMultiplier(multiplier);
        }}
        tooltipText="Apply auto vertical layout"
        defaultInputValue={spacementMultiplier}
        Icon={ArrowDownward}
        sliderOpen={slidersState.vertical}
        setSliderState={() =>
          setSlidersState((prev) => ({
            horizontal: false,
            vertical: !prev.vertical,
          }))
        }
      />
      <CustomAutoLayoutButton
        onAction={(multiplier) => {
          autoLayout.horizontal(multiplier);
          setSpacementMultiplier(multiplier);
        }}
        tooltipText="Apply auto horizontal layout"
        defaultInputValue={spacementMultiplier}
        Icon={ArrowForward}
        sliderOpen={slidersState.horizontal}
        setSliderState={() =>
          setSlidersState((prev) => ({
            vertical: false,
            horizontal: !prev.horizontal,
          }))
        }
      />
      <Tooltip
        title={
          allNodesExpanded ? 'Close all components' : 'Open all components'
        }
        placement="right"
      >
        <ControlButton
          about={
            allNodesExpanded ? 'Close all components' : 'Open all components'
          }
          onClick={() => {
            if (allNodesExpanded) contractNodes();
            else expandNodes();
            setAllNodesExpanded((val) => !val);
          }}
        >
          {allNodesExpanded ? <CloseFullscreenIcon /> : <OpenInFullIcon />}
        </ControlButton>
      </Tooltip>
    </Controls>
  );
};

export default TorchModelEditorControls;
