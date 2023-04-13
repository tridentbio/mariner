export const uniqBy = <T>(value: T[], get: (t: T) => string | number) => {
  const result: T[] = [];
  const memo = {};

  for (const v of value) {
    const k = get(v);

    if (k in memo) continue;
    result.push(v);
  }

  return result;
};

export const flatten = <T>(thing: T[][]): T[] => {
  return thing.reduce((acc, cur) => {
    return acc.concat(cur);
  }, []);
};

export const isArray = <T>(thing: T | T[]): thing is T[] => {
  return (
    Array.isArray(thing) &&
    thing &&
    typeof thing === 'object' &&
    'length' in thing
  );
};

export const range = (start: number, end: number) => {
  return new Array(end - start + 1).fill(0).map((_, idx) => start + idx);
};

export const sum = (numbers: number[]) =>
  numbers.reduce((acc, n) => acc + n, 0);

export const isArrayOfNumbers = (value: any) => {
  return (
    Array.isArray(value) &&
    !!value.length &&
    value.every((item) => !isNaN(Number(item)))
  );
};

export const replaceArrayLastValue = (targetArray: any[], newValue: any) => {
  targetArray.splice(-1, 1, newValue);
};
