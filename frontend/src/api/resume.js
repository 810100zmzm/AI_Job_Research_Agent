import request from './request'

export function getResumeOptions() {
  return request.get('/resume/options')
}

export function polishResume(payload) {
  return request.post('/resume/polish', payload)
}

export function getResumeFile(resumeId, format) {
  return request.get(`/resume/${encodeURIComponent(resumeId)}`, {
    params: { format },
    responseType: 'text',
    transformResponse: [(data) => data],
  })
}
