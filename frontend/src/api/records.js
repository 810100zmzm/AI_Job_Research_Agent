import request from './request'

export function createRecord(payload) {
  return request.post('/records', payload)
}

export function listRecords(status = '') {
  return request.get('/records', { params: status ? { status } : {} })
}

export function updateRecord(recordId, payload) {
  return request.put(`/records/${encodeURIComponent(recordId)}`, payload)
}

export function deleteRecord(recordId) {
  return request.delete(`/records/${encodeURIComponent(recordId)}`)
}
