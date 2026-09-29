import request from './request'

export function uploadFile(file, onUploadProgress) {
  const form = new FormData()
  form.append('file', file)

  return request.post('/upload', form, {
    onUploadProgress,
  })
}
