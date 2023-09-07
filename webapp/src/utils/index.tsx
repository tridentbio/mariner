import { ColumnConfig } from '@app/rtk/generated/models';
import api from 'app/api';
import { ColumnMeta, DataTypeGuard } from 'app/types/domain/datasets';
import { range } from './arrays';

export type ArrayElement<T> = T extends Array<infer C> ? C : never;
export type NonUndefined<T> = T extends undefined ? never : T;

export const isDev = () => {
  return import.meta.env.NODE_ENV === 'development';
};

/**
 * Format bytes as human-readable text.
 *
 * @param bytes Number of bytes.
 * @param si True to use metric (SI) units, aka powers of 1000. False to use
 *           binary (IEC), aka powers of 1024.
 * @param dp Number of decimal places to display.
 *
 * @return Formatted string.
 */
export const humanFileSize = (bytes: number, si = false, dp = 1) => {
  const thresh = si ? 1000 : 1024;

  if (Math.abs(bytes) < thresh) {
    return bytes + ' B';
  }

  const units = si
    ? ['kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    : ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
  let u = -1;
  const r = 10 ** dp;

  do {
    bytes /= thresh;
    ++u;
  } while (
    Math.round(Math.abs(bytes) * r) / r >= thresh &&
    u < units.length - 1
  );

  return bytes.toFixed(dp) + ' ' + units[u];
};

export const reverseString = (str: string): string => {
  const chars = str.split('');

  chars.reverse();

  return chars.join('');
};

export const splitLast = (
  str: string,
  sep: string
): readonly [string, string] => {
  const lastSepIndex = str.length - reverseString(str).indexOf(sep) - 1;

  return [
    str.substring(0, lastSepIndex),
    str.substring(lastSepIndex + 1),
  ] as const;
};

export const substrAfterLast = (str: string, sep: string): string => {
  const [_before, after] = splitLast(str, sep);

  return after;
};

export const keys = <T extends {}>(obj: T): (keyof T)[] => {
  return Object.keys(obj) as (keyof T)[];
};

export const randomLowerCase = (size: number = 16) => {
  return range(0, size - 1)
    .map(() => {
      const bounds = 'az';
      const start = bounds.charCodeAt(0);
      const end = bounds.charCodeAt(1);
      const randomChar = Math.floor(Math.random() * (end - start) + start);

      return String.fromCharCode(randomChar);
    })
    .join('');
};

export const makeS3DataLink = (datasetId: number, withErrors: boolean) =>
  `api/v1/datasets/${datasetId}/file${withErrors ? '-with-errors' : ''}`;

export const linkToBlob = async (link: string): Promise<File> => {
  const response = await api.get(link, { responseType: 'blob' });
  const metadata = {
    type: 'text/csv',
  };
  const file = new File([response.data], link, metadata);
  return file;
};

export const testPattern = (str: string) => (pattern: string) =>
  new RegExp(`^${pattern}$`).test(str);

export const testNonePattern = (patterns: string[]) => {
  return (val: string) => !patterns.some(testPattern(val));
};

export const isColumnUndescribed = (
  columnName: string,
  columnsMetadata: ColumnMeta[]
) => {
  return testNonePattern(columnsMetadata.map((col) => col.pattern))(columnName);
};

export const debounce = <T extends (...args: any[]) => any>(
  fn: T,
  ms: number = 700
) => {
  let timer: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

type Entries<T> = {
  [K in keyof T]: [K, T[K]];
}[keyof T][];
export const entries = <G,>(obj: G): Entries<G> => {
  // @ts-ignore
  return Object.entries(obj);
};

export const makeForm = (obj: {
  [key: string]: string | number | File | Blob;
}) => {
  const form = new FormData();
  Object.entries(obj).forEach(([key, value]) => {
    if (typeof value === 'string') form.set(key, value);
    else if (value instanceof File || value instanceof Blob) {
      form.set(key, value);
    }
  });
  return form;
};

type Grouped<T> = { [key: string]: T[] };
export const group = <T,>(
  arr: T[],
  keyGetter: (t: T) => string
): Grouped<T> => {
  const result: Record<string, T[]> = {};
  for (const element of arr) {
    const key = keyGetter(element);
    if (!(key in result)) result[key] = [];
    result[key].push(element);
  }
  return result;
};

export const title = (str: string) =>
  typeof str === 'string' && str
    ? str
        .split(' ')
        .map((s) => s[0].toUpperCase() + s.slice(1))
        .join(' ')
    : str;

export const findColumnMetadata = (
  descriptions: ColumnMeta[],
  colName: string
): ColumnMeta | undefined => {
  descriptions = [...descriptions].sort((a, b) =>
    b.pattern.length > a.pattern.length ? -1 : 1
  );
  return descriptions.find((description) => {
    return new RegExp(`^${description.pattern}$`).test(colName);
  });
};

export const isEven = (number: number) => !(number % 2);

export const cleanEmptyValues = <T extends Record<string, any>>(obj: T): T => {
  const keys = Object.keys(obj);
  const result = { ...obj };
  for (const key of keys) {
    if (obj[key] !== '') continue;
    delete result[key];
  }
  return result;
};

export const deepClone = (source: any[] | Record<string, any>) => {
  if (!source || typeof source !== 'object') return source;
  let clone: any[] | Record<string, any> = Array.isArray(source) ? [] : {};
  let value;
  for (const key in source) {
    value = source[key as keyof typeof source];
    clone[key as keyof typeof clone] =
      typeof value === 'object' ? deepClone(value) : value;
  }
  return clone;
};

const emailDomainRegex =
  /^@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])/gi;

export const isValidEmailDomain = (string: string) => {
  emailDomainRegex.lastIndex = 0;
  return emailDomainRegex.test(string);
};

export * from './arrays';

export type Required<T> = T extends undefined ? never : T;

export const defaultModeIsMax = (metricKey: string) => {
  const maxMetrics = ['ev', 'R2', 'pearson'];
  return maxMetrics.some((metric) => metricKey.includes(metric));
};

export const reprDataType = (dataType: ColumnConfig['dataType']) => {
  if (DataTypeGuard.isQuantity(dataType)) return `${dataType.unit}`;
  else if (DataTypeGuard.isCategorical(dataType)) return `Categorical`;
  else if (DataTypeGuard.isNumeric(dataType)) return `Numeric`;
  else if (DataTypeGuard.isSmiles(dataType)) return `SMILES`;
  else if (DataTypeGuard.isDna(dataType)) return `DNA`;
  else if (DataTypeGuard.isRna(dataType)) return `RNA`;
  else if (DataTypeGuard.isProtein(dataType)) return `Protein`;
  return `(${dataType.domainKind})`;
};

/**
 * Update value in structure by its path
 * @param path E.g. 'tables.table-name.columns.0'
 */
export const updateStructureByPath = (
  path: string,
  structure: object | Array<any>,
  value: any
): void => {
  let pathArray = path.split('.');

  const setStructure = (
    structure: object | Array<any>,
    path: string,
    value: any
  ) => {
    (structure[path as keyof typeof structure] as any) = value;
  };

  const getStructure = (structure: object | Array<any>, path: string) =>
    structure[path as keyof typeof structure];

  for (let i = 0; i < pathArray.length; i++) {
    const currentPath = pathArray[i];
    let currentStructure: any =
      structure[currentPath as keyof typeof structure];
    const isEndPath = !pathArray.length;

    if (!currentStructure && !isEndPath) {
      setStructure(structure, currentPath, {});
      currentStructure = getStructure(structure, currentPath);
    }

    if (currentStructure) {
      if (isEndPath) setStructure(structure, currentPath, value);
      else pathArray.shift();

      if (pathArray.length) {
        return updateStructureByPath(
          pathArray.join('.'),
          currentStructure,
          value
        );
      } else {
        setStructure(structure, currentPath, value);
      }
    }
  }
};
