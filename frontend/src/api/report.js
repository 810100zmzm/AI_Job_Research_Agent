import request from './request'

export function getReport(taskId, format) {
  return request.get(`/report/${encodeURIComponent(taskId)}`, {
    params: { format },
    responseType: 'text',
    transformResponse: [(data) => data],
  })
}
