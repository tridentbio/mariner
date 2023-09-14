const baseUrl = `${import.meta.env.VITE_API_BASE_URL}/api/v1`;

export const api = (path: string) => `${baseUrl}${path}`;
