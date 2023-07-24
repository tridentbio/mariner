import * as datasetsApi from 'app/rtk/generated/datasets';

type PossibleKeys =
  | 'mwt'
  | 'tpsa'
  | 'atom_count'
  | 'ring_count'
  | 'has_chiral_centers';
type HistValues = {
  values: (
    | {
        bin_start: number;
        bin_end: number;
        count: number;
      }
    | {
        min: number;
        max: number;
        mean: number;
        median: number;
      }
  )[];
};

export type Plot = {
  hist: HistValues;
};

export type PlotSmiles = Record<PossibleKeys, { hist: HistValues }>;

export type DataSummary = { [key: string]: Plot };

export type ColumnDescription = datasetsApi.ColumnsDescription;
export const enum DataTypeDomainKind {
  String = 'string',
  Numerical = 'numeric',
  Smiles = 'smiles',
  Categorical = 'categorical',
  Dna = 'dna',
  Rna = 'rna',
  Protein = 'protein',
}
export type DataType = datasetsApi.ColumnsDescription['dataType'];

export const DataTypeGuard = {
  isNumericalOrQuantity(
    dt: any
  ): dt is datasetsApi.NumericalDataType | datasetsApi.QuantityDataType {
    return dt.domainKind === DataTypeDomainKind.Numerical;
  },
  isNumeric(dt: any): dt is datasetsApi.NumericalDataType {
    return dt.domainKind === DataTypeDomainKind.Numerical && !('unit' in dt);
  },
  isQuantity(dt: any): dt is datasetsApi.QuantityDataType {
    return dt.domainKind === DataTypeDomainKind.Numerical && 'unit' in dt;
  },
  isCategorical(dt: any): dt is datasetsApi.CategoricalDataType {
    return dt.domainKind === DataTypeDomainKind.Categorical;
  },
  isString(dt: any): dt is datasetsApi.StringDataType {
    return dt.domainKind === DataTypeDomainKind.String;
  },
  isSmiles(dt: any): dt is datasetsApi.SmileDataType {
    return dt.domainKind === DataTypeDomainKind.Smiles;
  },
  isDna(dt: any): dt is datasetsApi.DnaDataType {
    return dt.domainKind === DataTypeDomainKind.Dna;
  },
  isRna(dt: any): dt is datasetsApi.RnaDataType {
    return dt.domainKind === DataTypeDomainKind.Rna;
  },
  isProtein(dt: any): dt is datasetsApi.ProteinDataType {
    return dt.domainKind === DataTypeDomainKind.Protein;
  },
};

export type ColumnMeta = datasetsApi.ColumnsDescription;
export type DatasetMetadata = {
  [key: string]: { [key2: string]: string | number };
};

export type DatasetsListingFilters = datasetsApi.GetMyDatasetsApiArg;

export type Dataset_ = datasetsApi.Dataset;

export interface DatasetErrors {
  columns: string[];
  rows: string[];
  logs: string[];
  dataset_error_key?: string;
}

export interface Dataset {
  id: number;
  name: string;
  description: string;
  rows: number;
  columns: number;
  bytes: number;
  stats: {
    train: DataSummary;
    val: DataSummary;
    test: DataSummary;
    full: DataSummary;
  };
  splitType: SplitType;
  splitTarget: string;
  splitActual: string;
  metadata: DatasetMetadata;
  createdAt: Date;
  updatedAt: Date;
  dataUrl: string;
  columnsMetadata?: ColumnMeta[];
  readyStatus: 'failed' | 'ready' | 'processing';
  errors?: DatasetErrors;
  createdById: number;
}

export type ColumnInfo = datasetsApi.ColumnsMeta;

export const enum SplitType {
  scaffold = 'scaffold',
  random = 'random',
}

export type NewDataset = datasetsApi.BodyCreateDatasetApiV1DatasetsPost;
export interface UpdateDataset extends NewDataset {
  datasetId: number;
}

export const isBioType = (type: DataTypeDomainKind): boolean => {
  return [
    DataTypeDomainKind.Dna,
    DataTypeDomainKind.Rna,
    DataTypeDomainKind.Protein,
  ].includes(type);
};
