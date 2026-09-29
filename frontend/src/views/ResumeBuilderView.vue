<script setup>
import { computed, ref } from 'vue'
import { DocumentAdd } from '@element-plus/icons-vue'

import ResumePolishPanel from '../components/ResumePolishPanel.vue'
import ResumeUpload from '../components/ResumeUpload.vue'
import { useSessionStore } from '../stores/session'

const session = useSessionStore()
const sourceMode = ref(session.currentTaskId ? 'task' : 'upload')
const resumeFileId = ref('')
const resumeBusy = ref(false)

const activeTaskId = computed(() => (sourceMode.value === 'task' ? session.currentTaskId : ''))
const activeResumeInput = computed(() => (sourceMode.value === 'upload' ? resumeFileId.value : ''))
const hasCurrentTask = computed(() => Boolean(session.currentTaskId))
const sourceReady = computed(() => Boolean(activeTaskId.value || activeResumeInput.value))
</script>

<template>
  <section class="resume-builder-page">
    <div class="page-heading">
      <div>
        <h1>生成简历</h1>
        <p>复用最近一次分析任务中的简历，或上传 / 选择一份简历后开始润色与生成。</p>
      </div>
      <el-tag :type="sourceReady ? 'success' : 'info'" effect="plain">
        {{ sourceReady ? '来源已就绪' : '等待简历来源' }}
      </el-tag>
    </div>

    <section class="surface source-panel">
      <div class="section-title">
        <h2>简历来源</h2>
        <span class="muted source-note">先选择来源，再选择表达强度和排版风格</span>
      </div>

      <el-radio-group v-model="sourceMode" class="source-mode" :disabled="resumeBusy">
        <el-radio-button value="task" :disabled="!hasCurrentTask">
          当前分析任务
        </el-radio-button>
        <el-radio-button value="upload">选择简历</el-radio-button>
      </el-radio-group>

      <div v-if="sourceMode === 'task' && hasCurrentTask" class="task-source">
        <el-icon class="source-icon"><DocumentAdd /></el-icon>
        <div>
          <strong>使用最近一次分析任务中的简历原文</strong>
          <span class="mono">{{ session.currentTaskId }}</span>
        </div>
      </div>

      <div v-else-if="sourceMode === 'upload'" class="upload-source">
        <ResumeUpload
          v-model="resumeFileId"
          :disabled="resumeBusy"
          @busy="resumeBusy = $event"
        />
      </div>

      <el-alert
        v-else
        type="info"
        title="还没有分析任务，请上传简历或从 input 选择。"
        :closable="false"
        show-icon
      />
    </section>

    <ResumePolishPanel
      :task-id="activeTaskId"
      :resume-input="activeResumeInput"
    />
  </section>
</template>

<style scoped>
.resume-builder-page {
  min-width: 0;
}

.source-panel {
  padding: 18px;
}

.source-note {
  font-size: 12px;
}

.source-mode {
  display: flex;
  flex-wrap: wrap;
}

.task-source {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
  padding: 14px;
  border: 1px solid #c7d7fe;
  border-radius: 8px;
  background: #f8faff;
}

.source-icon {
  color: #2563eb;
  font-size: 24px;
}

.task-source div {
  display: grid;
  min-width: 0;
  gap: 3px;
}

.task-source strong {
  color: #101828;
}

.task-source span {
  color: #667085;
  font-size: 12px;
  overflow-wrap: anywhere;
}

.upload-source {
  margin-top: 16px;
}

@media (max-width: 720px) {
  .source-mode,
  .source-mode :deep(.el-radio-button) {
    width: 100%;
  }

  .source-mode :deep(.el-radio-button__inner) {
    width: 100%;
  }
}
</style>
