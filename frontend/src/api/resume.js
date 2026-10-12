import request from './request'

export function getResumeOptions() {
  return request.get('/resume/options')
}

export function polishResume(payload) {
  // 润色可能分多次调用模型，单独放宽超时，避免浏览器先超时。
  return request.post('/resume/polish', payload, { timeout: 180000 })
}

export function getResumeFile(resumeId, format) {
  return request.get(`/resume/${encodeURIComponent(resumeId)}`, {
    params: { format },
    responseType: 'text',
    transformResponse: [(data) => data],
  })
}
