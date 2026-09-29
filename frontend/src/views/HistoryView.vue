<script setup>
import { onMounted, reactive, ref } from 'vue'
import { Delete, EditPen, Refresh } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import { useRecordsStore, STATUS_OPTIONS } from '../stores/records'

const recordsStore = useRecordsStore()
const loadError = ref('')
const editVisible = ref(false)
const saving = ref(false)
const formRef = ref(null)
const editingId = ref('')
const editForm = reactive({
  company: '',
  position: '',
  location: '',
  status: '待投',
  match_score: 0,
  note: '',
})

const editRules = {
  company: [{ required: true, message: '请输入公司名称', trigger: 'blur' }],
  position: [{ required: true, message: '请输入岗位名称', trigger: 'blur' }],
  status: [{ required: true, message: '请选择状态', trigger: 'change' }],
}
const statusType = {
  待投: 'info',
  已投: 'primary',
  笔试: 'warning',
  面试: 'warning',
  挂: 'danger',
  offer: 'success',
}

onMounted(load)

async function load(status = recordsStore.status) {
  loadError.value = ''
  try {
    await recordsStore.fetchRecords(status)
  } catch (error) {
    loadError.value = error.userMessage || '投递记录读取失败'
    ElMessage.error(loadError.value)
  }
}

async function remove(item) {
  try {
    await ElMessageBox.confirm(
      `确认删除「${item.company || '未填写公司'} · ${item.position || '未填写岗位'}」？`,
      '删除投递记录',
      {
        type: 'warning',
        confirmButtonText: '删除',
        cancelButtonText: '取消',
      },
    )
  } catch {
    return
  }

  try {
    await recordsStore.removeRecord(item.record_id)
    ElMessage.success('记录已删除')
  } catch (error) {
    ElMessage.error(error.userMessage || '删除失败')
  }
}

function openEdit(item) {
  editingId.value = item.record_id
  Object.assign(editForm, {
    company: item.company || '',
    position: item.position || '',
    location: item.location || '',
    status: item.status || '待投',
    match_score: Number(item.match_score || 0),
    note: item.note || '',
  })
  editVisible.value = true
}

async function saveEdit() {
  try {
    await formRef.value?.validate()
  } catch {
    return
  }

  saving.value = true
  try {
    await recordsStore.updateRecord(editingId.value, {
      company: editForm.company.trim(),
      position: editForm.position.trim(),
      location: editForm.location.trim(),
      status: editForm.status,
      match_score: Number(editForm.match_score || 0),
      note: editForm.note.trim(),
    })
    ElMessage.success('记录已更新')
    editVisible.value = false
  } catch (error) {
    ElMessage.error(error.userMessage || '更新失败')
  } finally {
    saving.value = false
  }
}

function score(item) {
  return Math.max(0, Math.min(100, Number(item.match_score || 0)))
}
</script>

<template>
  <section class="history-page">
    <div class="page-heading">
      <div>
        <h1>投递记录</h1>
        <p>{{ recordsStore.records.length }} 条记录</p>
      </div>
      <div class="filter-bar">
        <el-select
          v-model="recordsStore.status"
          placeholder="全部状态"
          clearable
          style="width: 160px"
          @change="load"
        >
          <el-option v-for="item in STATUS_OPTIONS" :key="item" :label="item" :value="item" />
        </el-select>
        <el-button :icon="Refresh" :loading="recordsStore.loading" @click="load()">刷新</el-button>
      </div>
    </div>

    <el-alert
      v-if="loadError"
      class="history-alert"
      type="error"
      :title="loadError"
      :closable="false"
      show-icon
    />

    <div class="surface table-surface">
      <el-table
        v-loading="recordsStore.loading"
        :data="recordsStore.records"
        height="100%"
        stripe
        border
        table-layout="fixed"
        empty-text="暂无投递记录"
        @row-dblclick="openEdit"
      >
        <el-table-column prop="company" label="公司" min-width="160" show-overflow-tooltip />
        <el-table-column prop="position" label="岗位" min-width="190" show-overflow-tooltip />
        <el-table-column prop="location" label="地点" width="110" show-overflow-tooltip />
        <el-table-column label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag :type="statusType[row.status] || 'info'" effect="light">{{ row.status || '待投' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="匹配分" width="150" align="center">
          <template #default="{ row }">
            <div v-if="Number(row.match_score) > 0" class="score-cell">
              <el-progress :percentage="score(row)" :stroke-width="7" :show-text="false" />
              <span>{{ score(row).toFixed(1) }}</span>
            </div>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
        <el-table-column prop="note" label="备注" min-width="220" show-overflow-tooltip />
        <el-table-column prop="updated_at" label="更新时间" width="165">
          <template #default="{ row }">
            <span class="mono time-cell">{{ row.updated_at ? new Date(row.updated_at * 1000).toLocaleString() : '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150" fixed="right" align="center">
          <template #default="{ row }">
            <el-button link type="primary" :icon="EditPen" @click="openEdit(row)">编辑</el-button>
            <el-button link type="danger" :icon="Delete" @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <el-dialog v-model="editVisible" title="编辑投递记录" width="min(620px, 92vw)">
      <el-form ref="formRef" :model="editForm" :rules="editRules" label-position="top">
        <div class="form-grid">
          <el-form-item label="公司" prop="company">
            <el-input v-model="editForm.company" />
          </el-form-item>
          <el-form-item label="岗位" prop="position">
            <el-input v-model="editForm.position" />
          </el-form-item>
          <el-form-item label="地点">
            <el-input v-model="editForm.location" />
          </el-form-item>
          <el-form-item label="状态" prop="status">
            <el-select v-model="editForm.status" style="width: 100%">
              <el-option v-for="item in STATUS_OPTIONS" :key="item" :label="item" :value="item" />
            </el-select>
          </el-form-item>
        </div>
        <el-form-item label="匹配分">
          <el-input-number v-model="editForm.match_score" :min="0" :max="100" :precision="1" style="width: 180px" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="editForm.note" type="textarea" :rows="3" resize="vertical" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="saveEdit">保存</el-button>
      </template>
    </el-dialog>
  </section>
</template>

<style scoped>
.history-page {
  width: 100%;
  display: flex;
  height: 100%;
  min-height: 0;
  min-width: 0;
  flex-direction: column;
  overflow: hidden;
}

.history-page > .page-heading {
  min-width: 0;
  flex: 0 0 auto;
  margin: 0;
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  background: #fff;
}

.filter-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.history-alert {
  margin: 14px 18px;
}

.table-surface {
  width: 100%;
  min-height: 0;
  min-width: 0;
  flex: 1;
  overflow: hidden;
  border-right: 0;
  border-bottom: 0;
  border-left: 0;
  border-radius: 0;
}

.table-surface :deep(.el-table) {
  height: 100%;
}

.score-cell {
  display: grid;
  grid-template-columns: minmax(60px, 1fr) 42px;
  align-items: center;
  gap: 8px;
  color: #344054;
  font-size: 12px;
}

.time-cell {
  color: #475467;
  font-size: 12px;
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0 14px;
}

@media (max-width: 680px) {
  .filter-bar {
    width: 100%;
  }

  .filter-bar .el-select {
    flex: 1;
  }

  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>
