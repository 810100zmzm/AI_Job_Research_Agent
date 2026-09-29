<script setup>
import { computed, ref, watch } from 'vue'
import { FolderOpened, UploadFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { uploadFile as uploadApi } from '../api/upload'
import InputFilePicker from './InputFilePicker.vue'

const props = defineProps({
  modelValue: { type: String, default: '' },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'busy'])
const mode = ref('upload')
const fileList = ref([])
const uploadedFileId = ref('')
const inputFileId = ref('')
const uploading = ref(false)
const inputBusy = ref(false)
const uploadPercent = ref(0)

const activeFileId = computed(() => (
  mode.value === 'input' ? inputFileId.value : uploadedFileId.value
))

watch(activeFileId, (value) => emit('update:modelValue', value), { immediate: true })
watch([uploading, inputBusy], ([upload, input]) => emit('busy', upload || input), { immediate: true })

async function handleFile(upload) {
  const file = upload.raw
  if (!file) return

  fileList.value = [upload]
  uploading.value = true
  uploadPercent.value = 0

  try {
    const result = await uploadApi(file, (event) => {
      if (event.total) uploadPercent.value = Math.round((event.loaded / event.total) * 100)
    })
    uploadedFileId.value = result.file_id
    ElMessage.success('简历已上传')
  } catch (error) {
    fileList.value = []
    uploadedFileId.value = ''
    ElMessage.error(error.userMessage || '简历上传失败')
  } finally {
    uploading.value = false
  }
}

function handleRemove() {
  fileList.value = []
  uploadedFileId.value = ''
  uploadPercent.value = 0
}
</script>

<template>
  <div class="resume-upload">
    <el-radio-group v-model="mode" :disabled="disabled || uploading || inputBusy" class="source-switch">
      <el-radio-button value="upload">
        <el-icon><UploadFilled /></el-icon>
        上传文件
      </el-radio-button>
      <el-radio-button value="input">
        <el-icon><FolderOpened /></el-icon>
        从 input 选择
      </el-radio-button>
    </el-radio-group>

    <InputFilePicker
      v-if="mode === 'input'"
      v-model="inputFileId"
      kind="resume"
      :disabled="disabled"
      @busy="inputBusy = $event"
    />

    <el-upload
      v-if="mode === 'upload'"
      drag
      action="#"
      accept=".md,.markdown,.txt,.text,.html,.htm,.xhtml,.json,.yaml,.yml"
      :auto-upload="false"
      :limit="1"
      :file-list="fileList"
      :disabled="disabled || uploading"
      :on-change="handleFile"
      :on-remove="handleRemove"
    >
      <el-icon class="upload-icon"><UploadFilled /></el-icon>
      <div class="upload-copy">拖拽简历到这里，或点击选择</div>
      <template #tip>
        <div class="upload-tip">支持 Markdown、TXT、HTML、JSON、YAML</div>
      </template>
    </el-upload>

    <el-progress
      v-if="mode === 'upload' && uploading"
      :percentage="uploadPercent"
      :stroke-width="6"
      :show-text="false"
    />
  </div>
</template>

<style scoped>
.source-switch {
  width: 100%;
}

.source-switch :deep(.el-radio-button) {
  flex: 1;
}

.source-switch :deep(.el-radio-button__inner) {
  display: inline-flex;
  width: 100%;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.resume-upload {
  display: grid;
  gap: 12px;
}

.resume-upload :deep(.el-upload),
.resume-upload :deep(.el-upload-dragger) {
  width: 100%;
}

.resume-upload :deep(.el-upload-dragger) {
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
</style>
