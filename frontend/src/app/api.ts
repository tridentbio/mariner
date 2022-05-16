
import axios from 'axios'
import { TOKEN } from './local-storage'

const api = axios.create({
  baseURL: process.env.API_BASE_URL,
})

api.interceptors.request.use(config => {
  const token = localStorage.get(TOKEN)
  if (!config || ! config.headers) return
  config.headers.Authorization = `Bearer ${token}`
  return config
})

export default api
