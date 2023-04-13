import { Box, IconButton, Tooltip } from '@mui/material';
import PhotoIcon from '@mui/icons-material/Photo';
import { useCallback } from 'react';
import { toSvg } from 'html-to-image';
import { ReactFlowInstance, useReactFlow } from 'react-flow-renderer';

type DownloadButtonProps = {
  tooltipText?: string;
  buttonDescription?: string;
  onClick?: (e: MouseEvent) => void;
};

const DownloadModelEditorImageButton: React.FC<DownloadButtonProps> = () => {
  const reactFlowInstance = useReactFlow();
  const exportModelToImage = useCallback(
    async (reactFlowInstance: ReactFlowInstance) => {
      const filter = (node: HTMLElement) => {
        const isControlsElement = node?.classList?.contains(
          'react-flow__controls'
        );
        const isMinimapElement = node?.classList?.contains(
          'react-flow__minimap'
        );
        return !(isControlsElement || isMinimapElement);
      };
      reactFlowInstance.fitView();
      const reactFlowElement = document.querySelector(
        '.react-flow__renderer'
      ) as HTMLElement;
      if (!reactFlowElement) return;

      const svgContent = await toSvg(reactFlowElement, {
        filter,
        canvasWidth: 50,
      });
      const svgElement = decodeURIComponent(
        svgContent.replace('data:image/svg+xml;charset=utf-8,', '').trim()
      );
      const svgBlob = new Blob([svgElement], {
        type: 'image/svg+xml;charset=utf-8',
      });
      const svgUrl = URL.createObjectURL(svgBlob);

      const downloadLink = document.createElement('a');
      downloadLink.href = svgUrl;
      downloadLink.download = 'newesttree.svg';
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
    },
    []
  );

  return (
    <Box
      sx={{
        width: 50,
        height: 50,
        borderRadius: '50%',
        position: 'absolute',
        right: 15,
        top: 15,
        zIndex: 200,
        boxShadow: 'rgba(0,0,0,0.24) 0px 3px 8px',
        backgroundColor: '#384E77',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Tooltip
        title="Click here to download the model graph image"
        placement="top"
      >
        <IconButton
          aria-label="Download graph image button"
          onClick={() => exportModelToImage(reactFlowInstance)}
        >
          <PhotoIcon sx={{ size: 'large', color: 'white' }} />
        </IconButton>
      </Tooltip>
    </Box>
  );
};

export { DownloadModelEditorImageButton };
