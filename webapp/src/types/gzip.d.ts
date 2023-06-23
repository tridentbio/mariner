declare module 'gzip-js' {
  export function zip(data: Uint8Array): Uint8Array;
  export function unzip(data: Uint8Array): Uint8Array;
}
