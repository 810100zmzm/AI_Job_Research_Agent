import { apiBaseURL } from './request'

export function createTraceStream(taskId, handlers = {}) {
  const url = `${apiBaseURL}/chat?task_id=${encodeURIComponent(taskId)}`
  const source = new EventSource(url)
  let closedByClient = false

  source.onopen = () => handlers.onOpen?.()

  source.onmessage = (event) => {
    try {
      handlers.onTrace?.(JSON.parse(event.data))
    } catch (error) {
      handlers.onError?.(new Error('Trace 数据格式错误'))
    }
  }

  source.addEventListener('done', (event) => {
    closedByClient = true
    source.close()
    handlers.onDone?.(parseEventData(event))
  })

  source.addEventListener('ask', (event) => {
    closedByClient = true
    source.close()
    handlers.onAsk?.(parseEventData(event))
  })

  source.addEventListener('error', (event) => {
    if (event.data) {
      closedByClient = true
      source.close()
      handlers.onError?.(new Error(parseEventData(event)?.detail || '分析任务失败'))
    }
  })

  source.onerror = () => {
    if (!closedByClient && source.readyState === EventSource.CLOSED) {
      handlers.onError?.(new Error('SSE 连接已断开'))
    }
  }

  return {
    close() {
      closedByClient = true
      source.close()
    },
  }
}

function parseEventData(event) {
  try {
    return JSON.parse(event.data)
  } catch {
    return null
  }
}
