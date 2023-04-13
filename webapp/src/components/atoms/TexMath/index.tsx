import { useLayoutEffect } from 'react';

const TexMath = ({ tex }: { tex: string }) => {
  // TODO: fix TextMath
  // MathJax is not working properly
  useLayoutEffect(() => {
    // @ts-ignore
    window.MathJax.Hub.Typeset();
  }, []);
  return (
    <span>
      <span className={'math'}>{tex}</span>
    </span>
  );
};
export default TexMath;
