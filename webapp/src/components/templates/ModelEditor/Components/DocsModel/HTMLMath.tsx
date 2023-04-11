import { useLayoutEffect } from 'react';

const HTMLMath = ({ html }: { html: string }) => {
  useLayoutEffect(() => {
    // @ts-ignore
    window.MathJax.Hub.Typeset();
  }, []);
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
};
export default HTMLMath;
