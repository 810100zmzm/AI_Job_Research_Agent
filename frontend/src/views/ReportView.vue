<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Back, DocumentAdd, RefreshRight } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { getReport } from '../api/report'
import ReportTabs from '../components/ReportTabs.vue'
import ResumePolishPanel from '../components/ResumePolishPanel.vue'
import { useRecordsStore, STATUS_OPTIONS } from '../stores/records'

const route = useRoute()
const router = useRouter()
const recordsStore = useRecordsStore()
const taskId = computed(() => String(route.params.taskId || ''))
const markdown = ref('')
const json = ref('')
const html = ref('')
const loading = ref(true)
const loadError = ref('')
const dialogVisible = ref(false)
const saving = ref(false)
const formRef = ref(null)

const form = reactive({
  company: '',
  position: '',
  location: '',
  status: '待投',
  match_score: 0,
  note: '',
})

const reportPayload = computed(() => {
  try {
    return JSON.parse(json.value || '{}')
  } catch {
    return {}
  }
})

const rules = {
  company: [{ required: true, message: '请输入公司名称', trigger: 'blur' }],
  position: [{ required: true, message: '请输入岗位名称', trigger: 'blur' }],
  status: [{ required: true, message: '请选择状态', trigger: 'change' }],
}

onMounted(loadReports)

async function loadReports() {
  loading.value = true
  loadError.value = ''
  try {
    const [md, jsonText, htmlText] = await Promise.all([
      getReport(taskId.value, 'md'),
      getReport(taskId.value, 'json'),
      getReport(taskId.value, 'html'),
    ])
    markdown.value = md || ''
    json.value = jsonText || ''
    html.value = htmlText || ''
  } catch (error) {
    loadError.value = error.userMessage || '报告读取失败'
  } finally {
    loading.value = false
  }
}

function openRecordDialog() {
  const payload = reportPayload.value
  form.company = payload.jd?.company || ''
  form.position = payload.jd?.title || ''
  form.location = ''
  form.status = '待投'
  form.match_score = 0
  form.note = payload.verdict?.headline || payload.question || ''
  dialogVisible.value = true
}

async function saveRecord() {
  try {
    await formRef.value?.validate()
  } catch {
    return
  }

  saving.value = true
  try {
    await recordsStore.addRecord({
      company: form.company.trim(),
      position: form.position.trim(),
      location: form.location.trim(),
      status: form.status,
      match_score: Number(form.match_score || 0),
      note: form.note.trim(),
    })
    ElMessage.success('已记入投递表')
    dialogVisible.value = false
  } catch (error) {
    ElMessage.error(error.userMessage || '保存投递记录失败')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <section>
    <div class="page-heading">
      <div>
        <h1>分析报告</h1>
        <p class="mono">{{ taskId }}</p>
      </div>
      <div class="heading-actions">
        <el-button :icon="Back" @click="router.push({ name: 'progress', params: { taskId } })">
          返回进度
        </el-button>
        <el-button type="primary" :icon="DocumentAdd" :disabled="loading || Boolean(loadError)" @click="openRecordDialog">
          记入投递表
        </el-button>
      </div>
    </div>

    <el-result
      v-if="loadError"
      icon="error"
      title="报告尚未就绪"
      :sub-title="loadError"
    >
      <template #extra>
        <el-button type="primary" :icon="RefreshRight" @click="loadReports">重新读取</el-button>
      </template>
    </el-result>

    <template v-else>
      <ReportTabs
        :markdown="markdown"
        :json="json"
        :html="html"
        :loading="loading"
      />
      <ResumePolishPanel v-if="!loading" :task-id="taskId" />
    </template>

    <el-dialog v-model="dialogVisible" title="记入投递表" width="min(620px, 92vw)">
      <el-form ref="formRef" :model="form" :rules="rules" label-position="top">
        <div class="form-grid">
          <el-form-item label="公司" prop="company">
            <el-input v-model="form.company" />
          </el-form-item>
          <el-form-item label="岗位" prop="position">
            <el-input v-model="form.position" />
          </el-form-item>
          <el-form-item label="地点">
            <el-input v-model="form.location" />
          </el-form-item>
          <el-form-item label="状态" prop="status">
            <el-select v-model="form.status" style="width: 100%">
              <el-option v-for="item in STATUS_OPTIONS" :key="item" :label="item" :value="item" />
            </el-select>
          </el-form-item>
        </div>
        <el-form-item label="匹配分">
          <el-input-number v-model="form.match_score" :min="0" :max="100" :precision="1" style="width: 180px" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="form.note" type="textarea" :rows="3" resize="vertical" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="saveRecord">保存</el-button>
      </template>
    </el-dialog>
  </section>
</template>

<style scoped>
.heading-actions {
  display: flex;
  gap: 10px;
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0 14px;
}

@media (max-width: 720px) {
  .heading-actions,
  .heading-actions .el-button {
    width: 100%;
  }

  .heading-actions {
    flex-direction: column;
  }

  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
