import { useEffect, useRef } from 'react';
import { SvgIconTypeMap, Tooltip, Typography } from '@mui/material';
import { OverridableComponent } from '@mui/material/OverridableComponent';
import { ControlButton } from 'reactflow';
import { marks } from './constants';

import { ButtonContainer, CustomSlider, InputContainer } from './styles';
type CustomAutoLayoutButtonProps = {
  tooltipText: string;
  onAction: (input: number) => void;
  defaultInputValue: number;
  Icon: OverridableComponent<SvgIconTypeMap<{}, 'svg'>> & {
    muiName: string;
  };
  sliderOpen: boolean;
  setSliderState: () => void;
};

const CustomAutoLayoutButton: React.FC<CustomAutoLayoutButtonProps> = ({
  onAction,
  tooltipText,
  defaultInputValue,
  Icon,
  sliderOpen,
  setSliderState,
}) => {
  const ButtonContainerRef = useRef<HTMLDivElement>(null);

  const handleClick = () => {
    if (!sliderOpen) onAction(defaultInputValue);
    setSliderState();
  };

  const handleInputChange = (value: number | number[]) => {
    onAction(value as number);
  };

  const handleClickOutsideSlider = (e: Event) => {
    const reference = ButtonContainerRef.current;
    if (!reference || !e.target) return;
    if (!reference.contains(e.target as HTMLElement)) {
      setSliderState();
    }
  };

  useEffect(() => {
    if (!sliderOpen) return;
    document.addEventListener('click', handleClickOutsideSlider);
    return () => {
      document.removeEventListener('click', handleClickOutsideSlider);
    };
  }, [sliderOpen]);

  return (
    <Tooltip title={tooltipText} placement="left" style={{ fontSize: '8px' }}>
      <ButtonContainer ref={ButtonContainerRef}>
        <ControlButton about={tooltipText} onClick={handleClick}>
          <Icon />
        </ControlButton>
        {sliderOpen ? (
          <InputContainer>
            <Typography
              sx={{ fontSize: 10, paddingLeft: 1 }}
              variant="subtitle2"
            >
              Space size
            </Typography>
            <CustomSlider
              aria-label="Restricted values"
              defaultValue={defaultInputValue}
              min={1}
              max={5}
              step={1}
              valueLabelDisplay="auto"
              marks={marks}
              sx={{ padding: 0 }}
              onChangeCommitted={(_e, value) => handleInputChange(value)}
            />
          </InputContainer>
        ) : null}
      </ButtonContainer>
    </Tooltip>
  );
};

export default CustomAutoLayoutButton;
