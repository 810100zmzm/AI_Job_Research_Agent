<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Back, Loading } from '@element-plus/icons-vue'

import { createTraceStream } from '../api/chat'
import { answerAnalysis } from '../api/analyze'
import AskResponse from '../components/AskResponse.vue'
import TraceStream from '../components/TraceStream.vue'
import { useSessionStore } from '../stores/session'

const route = useRoute()
const router = useRouter()
const session = useSessionStore()
const taskId = computed(() => String(route.params.taskId || ''))
const errorMessage = ref('')
const connected = ref(false)
const doneReceived = ref(false)
const waitingForAnswer = ref(false)
const question = ref('')
const questionReason = ref('')
const answerText = ref('')
const answering = ref(false)
let stream = null
let finishTimer = null

const statusLabel = computed(() => {
  if (errorMessage.value) return '任务失败'
  if (doneReceived.value) return '报告已生成'
  if (waitingForAnswer.value) return '等待补充'
  if (session.stopReceived) return '正在生成报告'
  if (connected.value) return '正在分析'
  return '连接中'
})

const isWorking = computed(
  () => !errorMessage.value && !doneReceived.value && !waitingForAnswer.value,
)

const statusType = computed(() => {
  if (errorMessage.value) return 'danger'
  if (doneReceived.value) return 'success'
  if (waitingForAnswer.value) return 'warning'
  return 'primary'
})

onMounted(() => {
  session.beginAnalysis(taskId.value)
  connect()
})

onBeforeUnmount(() => {
  if (finishTimer) clearTimeout(finishTimer)
  stream?.close()
})

function connect() {
  if (finishTimer) clearTimeout(finishTimer)
  errorMessage.value = ''
  connected.value = false
  doneReceived.value = false
  waitingForAnswer.value = false
  session.resetTrace()

  try {
    stream = createTraceStream(taskId.value, {
      onOpen: () => {
        connected.value = true
      },
      onTrace: (step) => {
        session.appendTrace(step)
      },
      onAsk: (payload) => {
        waitingForAnswer.value = true
        question.value = payload?.question || ''
        questionReason.value = payload?.question_reason || ''
        session.setAnalysisState('waiting_for_answer')
      },
      onDone: () => {
        doneReceived.value = true
        waitingForAnswer.value = false
        session.setAnalysisState('completed')
        finishTimer = setTimeout(() => {
          router.replace({ name: 'report', params: { taskId: taskId.value } })
        }, 500)
      },
      onError: (error) => {
        errorMessage.value = error.message
        waitingForAnswer.value = false
        session.setAnalysisState('failed')
      },
    })
  } catch (error) {
    errorMessage.value = error.message || '无法连接 SSE'
  }
}

async function submitAnswer() {
  const value = answerText.value.trim()
  if (!value || answering.value) return

  answering.value = true
  errorMessage.value = ''
  try {
    await answerAnalysis(taskId.value, value)
    answerText.value = ''
    question.value = ''
    questionReason.value = ''
    waitingForAnswer.value = false
    connect()
  } catch (error) {
    errorMessage.value = error.userMessage || error.message || '回答提交失败'
  } finally {
    answering.value = false
  }
}
</script>

<template>
  <section>
    <div class="page-heading">
      <div>
        <h1>分析进度</h1>
        <p class="mono">{{ taskId }}</p>
      </div>
      <el-tag :type="statusType" effect="light">
        <el-icon v-if="isWorking" class="is-loading"><Loading /></el-icon>
        <span class="status-dot" />
        {{ statusLabel }}
      </el-tag>
    </div>

    <div class="progress-summary surface">
      <div>
        <span class="muted">当前轮数</span>
        <strong>第 {{ session.currentRound }} 轮</strong>
      </div>
      <div>
        <span class="muted">Trace 步数</span>
        <strong>{{ session.trace.length }}</strong>
      </div>
      <div>
        <span class="muted">会话</span>
        <strong class="mono">{{ session.sessionId.slice(0, 18) }}</strong>
      </div>
      <el-button :icon="Back" @click="router.push('/')">返回输入</el-button>
    </div>

    <el-alert
      v-if="errorMessage"
      class="progress-alert"
      type="error"
      :title="errorMessage"
      :closable="false"
      show-icon
    />

    <AskResponse
      v-if="waitingForAnswer"
      v-model="answerText"
      :question="question"
      :question-reason="questionReason"
      :submitting="answering"
      @submit="submitAnswer"
    />

    <section class="surface trace-panel">
      <div class="section-title">
        <h2>实时 Trace</h2>
        <span class="muted">{{ session.trace.length }} 步</span>
      </div>
      <TraceStream :steps="session.trace" />
    </section>
  </section>
</template>

<style scoped>
.progress-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr)) auto;
  align-items: center;
  gap: 20px;
  margin-bottom: 16px;
  padding: 14px 18px;
}

.progress-summary > div {
  display: grid;
  gap: 2px;
}

.progress-summary strong {
  color: #101828;
  font-size: 16px;
  overflow-wrap: anywhere;
}

.progress-alert {
  margin-bottom: 16px;
}

.trace-panel {
  padding: 18px;
}

@media (max-width: 820px) {
  .progress-summary {
    grid-template-columns: 1fr 1fr;
  }

  .progress-summary .el-button {
    grid-column: 1 / -1;
    width: 100%;
  }
}
</style>
