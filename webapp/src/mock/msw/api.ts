const baseUrl = `${import.meta.env.VITE_API_BASE_URL}/v1`;

export const api = (path: string) => `${baseUrl}${path}`;
