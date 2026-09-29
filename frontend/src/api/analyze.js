import request from './request'

export function startAnalysis(payload) {
  return request.post('/analyze', payload)
}

export function answerAnalysis(taskId, answer) {
  return request.post(`/task/${encodeURIComponent(taskId)}/answer`, { answer })
}
