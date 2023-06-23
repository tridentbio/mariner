import { zip, unzip } from 'gzip-js';

// Library used: https://github.com/beatgammit/gzip-js

export const gzipCompress = async (data: File | null): Promise<File> => {
  if (data) {
    const reader = new FileReader();
    reader.readAsArrayBuffer(data);
    return new Promise((resolve, reject) => {
      reader.onload = () => {
        const compressed = new Uint8Array(
          zip(new Uint8Array(reader.result as ArrayBuffer))
        );
        const file = new File([compressed], `${data.name}.gz`, {
          type: 'application/gzip',
        });
        resolve(file);
      };
      reader.onerror = (error) => {
        reject(error);
      };
    });
  }
  return Promise.reject('No file provided');
};

export const gzipDecompress = async (data: Blob | null): Promise<Blob> => {
  if (data) {
    const reader = new FileReader();
    reader.readAsArrayBuffer(data);
    return new Promise((resolve, reject) => {
      reader.onload = () => {
        try {
          const decompressed = new Uint8Array(
            unzip(new Uint8Array(reader.result as ArrayBuffer))
          );
          const file = new Blob([decompressed], { type: 'text/csv' });
          resolve(file);
        } catch (e) {
          if (`${e}`.includes('Not a GZIP file')) resolve(data);
          reject(e);
        }
      };
      reader.onerror = (error) => {
        reject(error);
      };
    });
  }
  return Promise.reject('No file provided');
};
