<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { FolderOpened, Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { listInputFiles, selectInputFile } from '../api/inputFiles'

const props = defineProps({
  modelValue: { type: String, default: '' },
  kind: { type: String, required: true },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'busy', 'selected'])
const files = ref([])
const loading = ref(false)
const importing = ref(false)
const selectedPath = ref('')
const selectedLabel = ref('')

const hasFiles = computed(() => files.value.length > 0)
const busy = computed(() => loading.value || importing.value)

watch(busy, (value) => emit('busy', value), { immediate: true })

async function loadFiles() {
  loading.value = true
  try {
    files.value = await listInputFiles(props.kind)
  } catch (error) {
    files.value = []
    ElMessage.error(error.userMessage || 'input 文件读取失败')
  } finally {
    loading.value = false
  }
}

async function handleSelect(path) {
  selectedPath.value = path
  if (!path) {
    selectedLabel.value = ''
    emit('update:modelValue', '')
    return
  }

  importing.value = true
  try {
    const result = await selectInputFile(props.kind, path)
    selectedLabel.value = result.filename
    emit('update:modelValue', result.file_id)
    emit('selected', result)
    ElMessage.success('已读取 input 文件')
  } catch (error) {
    selectedPath.value = ''
    selectedLabel.value = ''
    emit('update:modelValue', '')
    ElMessage.error(error.userMessage || 'input 文件读取失败')
  } finally {
    importing.value = false
  }
}

function formatSize(size) {
  const value = Number(size || 0)
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

onMounted(loadFiles)
</script>

<template>
  <div class="input-file-picker">
    <div class="picker-row">
      <el-select
        v-model="selectedPath"
        filterable
        clearable
        :loading="loading"
        :disabled="disabled || importing"
        placeholder="选择 input 文件"
        class="picker-select"
        @change="handleSelect"
        @clear="handleSelect('')"
      >
        <template #prefix>
          <el-icon><FolderOpened /></el-icon>
        </template>
        <el-option
          v-for="item in files"
          :key="item.path"
          :label="item.label"
          :value="item.path"
        >
          <div class="option-row">
            <span class="option-name">{{ item.label }}</span>
            <span class="option-size">{{ formatSize(item.size) }}</span>
          </div>
        </el-option>
      </el-select>
      <el-tooltip content="刷新 input 文件" placement="top">
        <el-button
          :icon="Refresh"
          :loading="loading"
          :disabled="disabled || importing"
          circle
          @click="loadFiles"
        />
      </el-tooltip>
    </div>

    <div v-if="selectedLabel" class="selected-file">
      <el-icon><FolderOpened /></el-icon>
      <span>{{ selectedLabel }}</span>
    </div>
    <el-empty v-else-if="!loading && !hasFiles" description="input 目录没有可选文件" :image-size="54" />
  </div>
</template>

<style scoped>
.input-file-picker {
  display: grid;
  gap: 10px;
}

.picker-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 8px;
}

.picker-select {
  width: 100%;
}

.option-row {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}

.option-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.option-size {
  color: #98a2b3;
  font-size: 12px;
}

.selected-file {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
  padding: 9px 12px;
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  background: #f9fafb;
  color: #344054;
}

.selected-file span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
