
import axios from 'axios'
import { TOKEN } from './local-storage'

const api = axios.create({
  // TODO: fix env loading
  //baseURL: process.env.API_BASE_URL,
  baseURL: 'http://localhost/api'
})

api.interceptors.request.use(config => {
  const token = localStorage.getItem(TOKEN)
  if (!config || ! config.headers) return
  config.headers.Authorization = `Bearer ${token}`
  return config
})

export default api
