import { Selection } from 'seqviz/dist/selectionContext';
import { Highlight } from 'react-highlight-within-textarea';
import { BiologicalInputProps } from '.';

export const maskSeq = (seq: string) => {
  if (seq.length < 10) return seq;
  const matchs = [...seq.replace(/\s/g, '').matchAll(/\S{1,10}/g)];
  return matchs.join(' ');
};

export const getRange = (selected: Selection | null): Highlight[] => {
  if (!selected) return [];
  const { start, end, length, clockwise } = selected as {
    start: number;
    end: number;
    length: number;
    clockwise: boolean;
  };
  const ranges: Array<number[]> = [];
  if ((start > end && clockwise) || (start < end && !clockwise))
    ranges.push([0, end], [start, start + length - end]);
  else ranges.push([start, end].sort((a, b) => a - b));

  const result = ranges.map((range) => ({
    highlight: range,
    className: 'yellow',
  }));

  return result as Highlight[];
};

export const getRulePattern = (
  domainKind: BiologicalInputProps['domainKind']
) => {
  const rules = {
    dna: ['A', 'C', 'G', 'T'],
    rna: ['A', 'C', 'G', 'U'],
    // prettier-ignore
    protein: [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
      ],
  }[domainKind];
  return new RegExp(`[^${rules.join('')}\\s]`, 'gi');
};
