import { SeqViz, SeqVizProps } from 'seqviz';
import { Selection } from 'seqviz/dist/selectionContext';
import { useEffect, useMemo, useState } from 'react';
import {
  HighlightWithinTextarea,
  Highlight,
} from 'react-highlight-within-textarea';
import { Box } from '@mui/system';
import { FormLabel, Radio } from '@mui/material';
import { generateSeqvizSx, inputContainerSx, SeqContainer } from './styles';
import { getRange, getRulePattern } from './utils';

export type BiologicalInputProps = {
  onChange: (value: string) => void;
  value: string;
  key: string;
  label: string;
  domainKind: 'dna' | 'rna' | 'protein';
};

export const BiologicalInput = (props: BiologicalInputProps) => {
  const [selection, setSelection] = useState<Selection | null>(null);
  const [range, setRange] = useState<Highlight[]>([]);
  // const [maskedValue, setMaskedValue] = useState<string>('');
  const [translations, setTranslations] = useState<SeqVizProps['translations']>(
    []
  );
  const [visualization, setVisualization] = useState<'linear' | 'circular'>(
    'linear'
  );

  useEffect(() => {
    setRange(getRange(selection));
  }, [selection]);

  useEffect(() => {
    // prettier-ignore
    const notDnaRna = [
      'D','E','F','H','I','K','L','M',
      'N','P','Q','R','S','V','W','Y',
    ];
    if (notDnaRna.some((char) => props.value.toUpperCase().includes(char)))
      setTranslations([{ start: 0, end: 500, direction: 1 }]);
    else setTranslations([]);
  }, [props.value]);

  const rulePattern = useMemo(
    () => getRulePattern(props.domainKind),
    [props.domainKind]
  );

  const seqvizSx = useMemo(
    () => generateSeqvizSx(visualization),
    [visualization]
  );

  const seqType = useMemo(
    () =>
      ({
        dna: 'dna',
        rna: 'rna',
        protein: 'aa',
      }[props.domainKind] as SeqVizProps['seqType']),
    [props.domainKind]
  );

  return (
    <>
      <FormLabel id="demo-radio-buttons-group-label">{props.label}:</FormLabel>
      <SeqContainer>
        <Box
          sx={{
            width: '45vw',
            mt: '-0.8rem',
          }}
        >
          Visualization:
          <Radio
            checked={visualization === 'linear'}
            onChange={() => setVisualization('linear')}
          />
          Linear
          <Radio
            checked={visualization === 'circular'}
            onChange={() => setVisualization('circular')}
          />
          Circular
          <Box sx={inputContainerSx}>
            <HighlightWithinTextarea
              highlight={[
                ...range,
                {
                  highlight: rulePattern,
                  className: 'red',
                },
              ]}
              placeholder="Enter sequence..."
              value={props.value}
              onChange={(newValue) => {
                props.onChange(newValue.replace(/\s/g, ''));
                // TODO: fix masked value bug
                // setMaskedValue(maskSeq(newValue));
              }}
            />
          </Box>
        </Box>
        <Box sx={seqvizSx}>
          <SeqViz
            name={''}
            seqType={seqType}
            seq={props.value ? props.value.toUpperCase() : ''}
            translations={translations}
            zoom={{
              linear: 70,
            }}
            viewer={visualization}
            style={{
              width: '100%',
              height: '100%',
            }}
            onSelection={(selected) => setSelection(selected)}
          />
        </Box>
      </SeqContainer>
    </>
  );
};
