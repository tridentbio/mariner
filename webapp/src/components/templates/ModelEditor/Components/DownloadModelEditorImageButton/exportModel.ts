import { toSvg } from 'html-to-image';

export async function exportModelToImage() {
  const filter = (node: HTMLElement) => {
    const isControlsElement = node?.classList?.contains('react-flow__controls');
    const isMinimapElement = node?.classList?.contains('react-flow__minimap');
    return !(isControlsElement || isMinimapElement);
  };
  //get only printable elements by className
  const reactFlowElement = document.querySelector(
    '.react-flow__renderer'
  ) as HTMLElement;
  if (!reactFlowElement) return;

  const svgContent = await toSvg(reactFlowElement, { filter, canvasWidth: 50 });
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
}
