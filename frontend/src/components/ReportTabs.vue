<script setup>
import { computed, ref } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'

const props = defineProps({
  markdown: { type: String, default: '' },
  json: { type: String, default: '' },
  html: { type: String, default: '' },
  loading: { type: Boolean, default: false },
})

const activeTab = ref('markdown')

const markdownHtml = computed(() => {
  if (!props.markdown) return ''
  return DOMPurify.sanitize(marked.parse(props.markdown), { USE_PROFILES: { html: true } })
})

const safeHtml = computed(() => DOMPurify.sanitize(props.html || '', { USE_PROFILES: { html: true } }))

const formattedJson = computed(() => {
  if (!props.json) return ''
  try {
    return JSON.stringify(JSON.parse(props.json), null, 2)
  } catch {
    return props.json
  }
})
</script>

<template>
  <el-tabs v-model="activeTab" class="report-tabs">
    <el-tab-pane label="Markdown" name="markdown">
      <div v-loading="loading" class="report-pane markdown-body" v-html="markdownHtml" />
    </el-tab-pane>
    <el-tab-pane label="JSON" name="json">
      <pre v-loading="loading" class="report-pane json-body">{{ formattedJson }}</pre>
    </el-tab-pane>
    <el-tab-pane label="HTML" name="html">
      <div v-loading="loading" class="report-pane html-frame">
        <div class="html-body" v-html="safeHtml" />
      </div>
    </el-tab-pane>
  </el-tabs>
</template>

<style scoped>
.report-tabs :deep(.el-tabs__header) {
  margin-bottom: 0;
}

.report-pane {
  min-height: 420px;
  padding: 20px;
  border: 1px solid #e4e7ec;
  border-top: 0;
  border-radius: 0 0 8px 8px;
  background: #fff;
}

.markdown-body,
.html-body {
  color: #344054;
  line-height: 1.75;
  overflow-wrap: anywhere;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.html-body :deep(h1),
.html-body :deep(h2),
.html-body :deep(h3) {
  margin-top: 1.4em;
  color: #101828;
  line-height: 1.35;
}

.markdown-body :deep(table),
.html-body :deep(table) {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.markdown-body :deep(th),
.markdown-body :deep(td),
.html-body :deep(th),
.html-body :deep(td) {
  padding: 8px 10px;
  border: 1px solid #e4e7ec;
  text-align: left;
  vertical-align: top;
}

.markdown-body :deep(code) {
  padding: 2px 5px;
  border-radius: 4px;
  background: #f2f4f7;
  font-family: "Cascadia Code", Consolas, monospace;
}

.json-body {
  overflow: auto;
  margin: 0;
  color: #182230;
  background: #f8fafc;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.html-frame {
  padding: 0;
  overflow: hidden;
}

.html-body {
  min-height: 420px;
  padding: 20px;
  background: #fff;
}
</style>
