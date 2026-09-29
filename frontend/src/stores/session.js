import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

const STORAGE_KEY = 'job-research-session'

function readSession() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
  } catch {
    return {}
  }
}

function makeSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `web-${crypto.randomUUID()}`
  }
  return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export const useSessionStore = defineStore('session', () => {
  const saved = readSession()
  const sessionId = ref(saved.sessionId || makeSessionId())
  const currentTaskId = ref(saved.currentTaskId || '')
  const trace = ref([])
  const analysisState = ref(saved.currentTaskId ? 'running' : 'idle')

  const currentRound = computed(
    () => 1 + trace.value.filter((step) => step.action === 'ApplyUserAnswer').length,
  )
  const stopReceived = computed(() => trace.value.some((step) => step.decision === 'Stop'))

  function persist() {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ sessionId: sessionId.value, currentTaskId: currentTaskId.value }),
    )
  }

  function beginAnalysis(taskId) {
    currentTaskId.value = taskId
    trace.value = []
    analysisState.value = 'running'
    persist()
  }

  function appendTrace(step) {
    trace.value.push(step)
  }

  function setAnalysisState(state) {
    analysisState.value = state
  }

  function resetTrace() {
    trace.value = []
    analysisState.value = 'running'
  }

  return {
    sessionId,
    currentTaskId,
    trace,
    analysisState,
    currentRound,
    stopReceived,
    beginAnalysis,
    appendTrace,
    setAnalysisState,
    resetTrace,
  }
})
