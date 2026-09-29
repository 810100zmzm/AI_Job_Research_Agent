<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { Document, FolderOpened, Link, Picture, UploadFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { uploadFile as uploadApi } from '../api/upload'
import InputFilePicker from './InputFilePicker.vue'

const props = defineProps({
  modelValue: { type: String, default: '' },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'busy'])
const mode = ref('text')
const text = ref('')
const url = ref('')
const fileId = ref('')
const inputFileId = ref('')
const filename = ref('')
const previewUrl = ref('')
const uploading = ref(false)
const inputBusy = ref(false)
const uploadPercent = ref(0)

const currentInput = computed(() => {
  if (mode.value === 'text') return text.value.trim()
  if (mode.value === 'url') return url.value.trim()
  if (mode.value === 'input') return inputFileId.value
  return fileId.value
})

watch(currentInput, (value) => emit('update:modelValue', value), { immediate: true })
watch([uploading, inputBusy], ([upload, input]) => emit('busy', upload || input), { immediate: true })

async function handleScreenshot(upload) {
  const file = upload.raw
  if (!file) return
  if (file.size > 10 * 1024 * 1024) {
    ElMessage.warning('截图不能超过 10 MB')
    return
  }

  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(file)
  filename.value = file.name
  uploading.value = true
  uploadPercent.value = 0

  try {
    const result = await uploadApi(file, (event) => {
      if (event.total) uploadPercent.value = Math.round((event.loaded / event.total) * 100)
    })
    fileId.value = result.file_id
    ElMessage.success('截图已上传')
  } catch (error) {
    fileId.value = ''
    ElMessage.error(error.userMessage || '截图上传失败')
  } finally {
    uploading.value = false
  }
}

function clearScreenshot() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
  filename.value = ''
  fileId.value = ''
  uploadPercent.value = 0
}

onBeforeUnmount(() => {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
})
</script>

<template>
  <div class="jd-input">
    <el-radio-group v-model="mode" :disabled="disabled" class="mode-switch">
      <el-radio-button value="text">
        <el-icon><Document /></el-icon>
        粘贴文本
      </el-radio-button>
      <el-radio-button value="screenshot">
        <el-icon><Picture /></el-icon>
        上传截图
      </el-radio-button>
      <el-radio-button value="input">
        <el-icon><FolderOpened /></el-icon>
        从 input 选择
      </el-radio-button>
      <el-radio-button value="url">
        <el-icon><Link /></el-icon>
        粘贴网址
      </el-radio-button>
    </el-radio-group>

    <el-input
      v-if="mode === 'text'"
      v-model="text"
      type="textarea"
      :rows="10"
      resize="vertical"
      :disabled="disabled"
      placeholder="粘贴岗位职责与任职要求"
    />

    <el-input
      v-else-if="mode === 'url'"
      v-model="url"
      :disabled="disabled"
      size="large"
      placeholder="https://example.com/jobs/123"
    >
      <template #prefix>
        <el-icon><Link /></el-icon>
      </template>
    </el-input>

    <div v-else-if="mode === 'input'" class="input-source">
      <InputFilePicker
        v-model="inputFileId"
        kind="jd"
        :disabled="disabled"
        @busy="inputBusy = $event"
      />
    </div>

    <div v-else class="screenshot-input">
      <el-upload
        drag
        action="#"
        accept="image/*"
        :auto-upload="false"
        :show-file-list="false"
        :disabled="disabled || uploading"
        :on-change="handleScreenshot"
      >
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-copy">拖拽 JD 截图到这里，或点击选择</div>
        <template #tip>
          <div class="upload-tip">PNG / JPG / WEBP，单张不超过 10 MB</div>
        </template>
      </el-upload>

      <div v-if="previewUrl" class="screenshot-preview">
        <img :src="previewUrl" :alt="filename || 'JD 截图预览'" />
        <div class="preview-meta">
          <span class="preview-name">{{ filename }}</span>
          <el-button link type="danger" :disabled="uploading" @click="clearScreenshot">移除</el-button>
        </div>
      </div>

      <el-progress
        v-if="uploading"
        :percentage="uploadPercent"
        :stroke-width="6"
        :show-text="false"
      />
    </div>
  </div>
</template>

<style scoped>
.jd-input {
  display: grid;
  gap: 14px;
}

.mode-switch {
  width: 100%;
}

.mode-switch :deep(.el-radio-button) {
  flex: 1;
}

.mode-switch :deep(.el-radio-button__inner) {
  display: inline-flex;
  width: 100%;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.input-source {
  min-height: 150px;
}

.screenshot-input {
  display: grid;
  gap: 12px;
}

.screenshot-input :deep(.el-upload),
.screenshot-input :deep(.el-upload-dragger) {
  width: 100%;
}

.screenshot-input :deep(.el-upload-dragger) {
  min-height: 190px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
}

.upload-icon {
  color: #2563eb;
  font-size: 34px;
}

.upload-copy {
  margin-top: 8px;
  color: #344054;
}

.upload-tip {
  margin-top: 4px;
  color: #667085;
  font-size: 12px;
}

.screenshot-preview {
  display: grid;
  grid-template-columns: 76px minmax(0, 1fr);
  align-items: center;
  gap: 12px;
  padding: 10px;
  border: 1px solid #e4e7ec;
  border-radius: 8px;
  background: #f9fafb;
}

.screenshot-preview img {
  width: 76px;
  height: 58px;
  object-fit: cover;
  border-radius: 6px;
  background: #eaecf0;
}

.preview-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  min-width: 0;
}

.preview-name {
  overflow: hidden;
  color: #344054;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (max-width: 560px) {
  .mode-switch {
    display: grid;
  }

  .mode-switch :deep(.el-radio-button__inner) {
    width: 100%;
  }
}
</style>
