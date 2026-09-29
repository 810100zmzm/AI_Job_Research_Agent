<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { Promotion } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { startAnalysis } from '../api/analyze'
import JDInput from '../components/JDInput.vue'
import ResumeUpload from '../components/ResumeUpload.vue'
import { useSessionStore } from '../stores/session'

const router = useRouter()
const session = useSessionStore()
const jdInput = ref('')
const resumeFileId = ref('')
const jdBusy = ref(false)
const resumeBusy = ref(false)
const submitting = ref(false)

const canSubmit = computed(
  () => Boolean(jdInput.value && resumeFileId.value) && !jdBusy.value && !resumeBusy.value && !submitting.value,
)

async function submit() {
  if (!jdInput.value) {
    ElMessage.warning('请提供 JD 文本、截图、网址或 input 文件')
    return
  }
  if (!resumeFileId.value) {
    ElMessage.warning('请上传简历或从 input 选择')
    return
  }

  submitting.value = true
  try {
    const result = await startAnalysis({
      jd_input: jdInput.value,
      resume_input: resumeFileId.value,
      session_id: session.sessionId,
      user_id: 'web-user',
    })
    session.beginAnalysis(result.task_id)
    await router.push({ name: 'progress', params: { taskId: result.task_id } })
  } catch (error) {
    ElMessage.error(error.userMessage || '创建分析任务失败')
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <section>
    <div class="page-heading">
      <div>
        <h1>分析输入</h1>
        <p>岗位信息与简历会提交到同一条尽调任务。</p>
      </div>
      <el-tag type="info" effect="plain" class="mono">
        {{ session.sessionId.slice(0, 18) }}
      </el-tag>
    </div>

    <div class="input-grid">
      <section class="surface input-panel">
        <div class="section-title">
          <h2>岗位信息</h2>
          <el-tag size="small" type="primary" effect="plain">JD</el-tag>
        </div>
        <JDInput
          v-model="jdInput"
          :disabled="submitting"
          @busy="jdBusy = $event"
        />
      </section>

      <section class="surface input-panel">
        <div class="section-title">
          <h2>简历文件</h2>
          <el-tag size="small" type="primary" effect="plain">Resume</el-tag>
        </div>
        <ResumeUpload
          v-model="resumeFileId"
          :disabled="submitting"
          @busy="resumeBusy = $event"
        />
      </section>
    </div>

    <div class="submit-bar surface">
      <div>
        <strong>准备状态</strong>
        <span class="muted">
          JD {{ jdInput ? '已提供' : '待提供' }} · 简历 {{ resumeFileId ? '已提供' : '待提供' }}
        </span>
      </div>
      <el-button
        type="primary"
        size="large"
        :icon="Promotion"
        :loading="submitting"
        :disabled="!canSubmit"
        @click="submit"
      >
        开始分析
      </el-button>
    </div>
  </section>
</template>

<style scoped>
.input-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
  gap: 16px;
}

.input-panel {
  min-width: 0;
  padding: 18px;
}

.submit-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  margin-top: 16px;
  padding: 14px 18px;
}

.submit-bar > div {
  display: grid;
  gap: 2px;
}

.submit-bar strong {
  color: #101828;
}

.submit-bar span {
  font-size: 13px;
}

@media (max-width: 900px) {
  .input-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 600px) {
  .submit-bar {
    align-items: stretch;
    flex-direction: column;
  }

  .submit-bar .el-button {
    width: 100%;
  }
}
</style>
