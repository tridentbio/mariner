import { ColumnMeta } from '@app/types/domain/datasets';

export const findMeta = (
  column: string,
  columnsMeta: ColumnMeta[]
): ColumnMeta | undefined => {
  return columnsMeta.find((meta) => {
    const regex = new RegExp(meta.pattern);
    if (regex.test(column)) return true;
    return false;
  });
};

export const datasetsGraphTitlesMapper = {
  mwt: 'molecular weight',
  tpsa: 'topological polar surface area',
  atom_count: 'atom count',
  ring_count: 'ring count',
  has_chiral_centers: 'contains chiral centers',
  sequence_lengh: 'sequence length',
  gc_content: 'GC content',
  gaps_number: 'number of gaps',
};
