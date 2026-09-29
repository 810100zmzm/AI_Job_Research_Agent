import request from './request'

export function listInputFiles(kind) {
  return request.get('/input-files', { params: { kind } })
}

export function selectInputFile(kind, path) {
  return request.post('/input-files/select', { kind, path })
}
