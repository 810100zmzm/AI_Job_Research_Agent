<script setup>
import { computed, onMounted, ref } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { Document, Download, MagicStick } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

import { getResumeOptions, polishResume } from '../api/resume'

const props = defineProps({
  taskId: { type: String, default: '' },
  resumeInput: { type: String, default: '' },
})

const hasSource = computed(() => Boolean(props.taskId || props.resumeInput))
const sourceHint = computed(() =>
  props.taskId ? '使用当前分析任务中的简历原文' : '使用上传的简历文件',
)

const optionsLoading = ref(false)
const polishing = ref(false)
const generating = ref(false)
const levels = ref([])
const styles = ref([])
const level = ref('L2')
const style = ref('classic')
const polishResult = ref(null)
const generateResult = ref(null)
const activeResult = ref('polish')
const activeTab = ref('compare')

const currentResult = computed(() =>
  activeResult.value === 'generate' ? generateResult.value : polishResult.value,
)
const hasBothResults = computed(() => Boolean(polishResult.value && generateResult.value))

const markdownHtml = computed(() => {
  if (!currentResult.value?.markdown) return ''
  return DOMPurify.sanitize(marked.parse(currentResult.value.markdown), { USE_PROFILES: { html: true } })
})

const safeHtml = computed(() => DOMPurify.sanitize(currentResult.value?.html || '', { USE_PROFILES: { html: true } }))

const levelLabel = computed(() => {
  const item = levels.value.find((entry) => entry.key === level.value)
  return item ? `${item.key} · ${item.name}` : level.value
})

const styleLabel = computed(() => {
  const item = styles.value.find((entry) => entry.key === style.value)
  return item?.name || style.value
})

onMounted(loadOptions)

async function loadOptions() {
  optionsLoading.value = true
  try {
    const payload = await getResumeOptions()
    levels.value = payload.levels || []
    styles.value = payload.styles || []
    if (!levels.value.some((item) => item.key === level.value)) level.value = levels.value[0]?.key || 'L2'
    if (!styles.value.some((item) => item.key === style.value)) style.value = styles.value[0]?.key || 'classic'
  } catch (error) {
    ElMessage.error(error.userMessage || '简历选项读取失败')
  } finally {
    optionsLoading.value = false
  }
}

async function runResumeAction(action) {
  if (!hasSource.value) {
    ElMessage.warning('请先提供简历来源')
    return
  }

  const isPolish = action === 'polish'
  if (isPolish) polishing.value = true
  else generating.value = true

  try {
    const payload = await polishResume({
      task_id: props.taskId,
      resume_input: props.resumeInput,
      level: level.value,
      style: style.value,
      enable_polish: isPolish,
    })

    if (isPolish) {
      polishResult.value = payload
      activeResult.value = 'polish'
      activeTab.value = 'compare'
      ElMessage.success('原文对照与润色版简历已生成')
    } else {
      generateResult.value = payload
      activeResult.value = 'generate'
      activeTab.value = 'markdown'
      ElMessage.success('原文简历已生成')
    }
  } catch (error) {
    ElMessage.error(error.userMessage || (isPolish ? '简历润色失败' : '简历生成失败'))
  } finally {
    if (isPolish) polishing.value = false
    else generating.value = false
  }
}

function switchResult(mode) {
  activeResult.value = mode
  activeTab.value = mode === 'polish' ? 'compare' : 'markdown'
}

function polish() {
  return runResumeAction('polish')
}

function generate() {
  return runResumeAction('generate')
}

function download(format) {
  const content = format === 'html' ? currentResult.value?.html : currentResult.value?.markdown
  if (!content) return
  const type = format === 'html' ? 'text/html;charset=utf-8' : 'text/markdown;charset=utf-8'
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  const mode = activeResult.value === 'generate' ? 'original' : currentResult.value.level.toLowerCase()
  link.download = `resume-${mode}-${currentResult.value.style}.${format}`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
}
</script>

<template>
  <section class="surface resume-builder">
    <div class="builder-head">
      <div>
        <h2>润色与生成简历</h2>
        <p>{{ sourceHint }}，可分别执行润色或生成</p>
      </div>
      <div class="builder-actions">
        <el-button
          v-if="currentResult"
          :icon="Download"
          @click="download('md')"
        >
          Markdown
        </el-button>
        <el-button
          v-if="currentResult"
          :icon="Download"
          @click="download('html')"
        >
          HTML
        </el-button>
        <el-button
          :icon="MagicStick"
          :loading="polishing"
          :disabled="!hasSource || generating || optionsLoading"
          @click="polish"
        >
          开始润色
        </el-button>
        <el-button
          type="primary"
          :icon="Document"
          :loading="generating"
          :disabled="!hasSource || polishing || optionsLoading"
          @click="generate"
        >
          生成简历
        </el-button>
      </div>
    </div>

    <div class="builder-controls">
      <div class="control-block">
        <span class="control-label">表达强度</span>
        <el-radio-group v-model="level" :disabled="polishing || generating">
          <el-radio-button v-for="item in levels" :key="item.key" :value="item.key">
            {{ item.key }} · {{ item.name }}
          </el-radio-button>
        </el-radio-group>
      </div>
      <div class="control-block style-block">
        <span class="control-label">简历风格</span>
        <el-select v-model="style" :disabled="polishing || generating" style="width: 190px">
          <el-option
            v-for="item in styles"
            :key="item.key"
            :label="item.name"
            :value="item.key"
          />
        </el-select>
      </div>
      <div v-if="hasBothResults" class="result-switch">
        <span class="control-label">查看结果</span>
        <el-radio-group :model-value="activeResult" size="small" @change="switchResult">
          <el-radio-button value="polish">润色结果</el-radio-button>
          <el-radio-button value="generate">生成结果</el-radio-button>
        </el-radio-group>
      </div>
    </div>

    <template v-if="currentResult">
      <el-alert
        v-if="currentResult.notice"
        class="builder-notice"
        :type="currentResult.stats.original ? 'warning' : 'info'"
        :title="currentResult.notice"
        :closable="false"
        show-icon
      />

      <div v-if="activeResult === 'polish'" class="builder-stats">
        <el-tag effect="plain">条目 {{ currentResult.stats.total }}</el-tag>
        <el-tag type="success" effect="plain">规则词典 {{ currentResult.stats.dictionary }}</el-tag>
        <el-tag type="primary" effect="plain">LLM {{ currentResult.stats.llm }}</el-tag>
        <el-tag type="warning" effect="plain">原文保留 {{ currentResult.stats.original }}</el-tag>
        <el-tag type="info" effect="plain">被挡事实 {{ currentResult.stats.blocked }}</el-tag>
      </div>

      <el-tabs v-model="activeTab" class="builder-tabs">
        <el-tab-pane v-if="activeResult === 'polish'" label="原文对照" name="compare">
          <el-table :data="currentResult.items" stripe border table-layout="fixed" class="compare-table">
            <el-table-column prop="ref" label="出处" width="145" show-overflow-tooltip />
            <el-table-column prop="original" label="原文" min-width="280" show-overflow-tooltip />
            <el-table-column prop="polished" label="润色后" min-width="280" show-overflow-tooltip />
            <el-table-column label="来源" width="125" align="center">
              <template #default>
                <el-tag size="small" effect="plain" class="source-tag">llm-polish</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="status" label="处理" width="125" show-overflow-tooltip />
          </el-table>
        </el-tab-pane>
        <el-tab-pane label="Markdown" name="markdown">
          <div class="resume-preview markdown-body" v-html="markdownHtml" />
        </el-tab-pane>
        <el-tab-pane label="HTML" name="html">
          <div class="resume-preview html-body" v-html="safeHtml" />
        </el-tab-pane>
      </el-tabs>
    </template>

    <div v-else class="builder-empty">
      <el-empty :description="hasSource ? '可选择开始润色或直接生成简历' : '请先选择简历来源'" />
    </div>
  </section>
</template>

<style scoped>
.resume-builder {
  margin-top: 18px;
  padding: 18px;
}

.builder-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.builder-head h2 {
  margin: 0;
  color: #101828;
  font-size: 17px;
}

.builder-head p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 13px;
}

.builder-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.builder-actions .el-button + .el-button {
  margin-left: 0;
}

.builder-controls {
  display: flex;
  align-items: flex-end;
  flex-wrap: wrap;
  gap: 24px;
  margin-top: 18px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
}

.control-block,
.result-switch {
  display: grid;
  gap: 7px;
}

.result-switch {
  margin-left: auto;
}

.control-label {
  color: #475467;
  font-size: 12px;
  font-weight: 600;
}

.builder-notice {
  margin-top: 16px;
}

.builder-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

.builder-tabs {
  margin-top: 10px;
}

.compare-table {
  width: 100%;
}

.resume-preview {
  min-height: 420px;
  max-height: 720px;
  overflow: auto;
  padding: 20px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
}

.markdown-body,
.html-body {
  color: #344054;
  line-height: 1.72;
  overflow-wrap: anywhere;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.html-body :deep(h1),
.html-body :deep(h2),
.html-body :deep(h3) {
  color: #101828;
  line-height: 1.35;
}

.markdown-body :deep(table),
.html-body :deep(table) {
  width: 100%;
  border-collapse: collapse;
}

.markdown-body :deep(th),
.markdown-body :deep(td),
.html-body :deep(th),
.html-body :deep(td) {
  padding: 7px 9px;
  border: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}

.source-tag {
  font-family: "Cascadia Code", Consolas, monospace;
}

.builder-empty {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--line);
}

@media (max-width: 840px) {
  .builder-head,
  .builder-controls {
    align-items: stretch;
    flex-direction: column;
  }

  .builder-actions {
    justify-content: flex-start;
  }

  .builder-actions .el-button {
    flex: 1;
  }

  .result-switch {
    margin-left: 0;
  }

  .style-block .el-select {
    width: 100% !important;
  }
}
</style>
