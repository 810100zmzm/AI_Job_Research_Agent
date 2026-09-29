import axios from 'axios'

export const apiBaseURL = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '')

const request = axios.create({
  baseURL: apiBaseURL,
  timeout: 60000,
})

request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail = error.response?.data?.detail
    const status = error.response?.status
    error.userMessage = detail || (status ? `请求失败（${status}）` : '网络连接失败，请确认后端服务已启动')
    return Promise.reject(error)
  },
)

export default request
