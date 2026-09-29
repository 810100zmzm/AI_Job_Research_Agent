<script setup>
import { nextTick, ref, watch } from 'vue'

const props = defineProps({
  steps: { type: Array, default: () => [] },
})

const scroller = ref(null)
const decisionTypes = {
  Continue: 'primary',
  Adjust: 'warning',
  Ask: 'warning',
  Stop: 'success',
}

watch(
  () => props.steps.length,
  async () => {
    await nextTick()
    if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight
  },
)
</script>

<template>
  <div ref="scroller" class="trace-stream" aria-live="polite">
    <el-skeleton v-if="!steps.length" :rows="5" animated />

    <article v-for="step in steps" :key="step.index || `${step.action}-${step.decision}`" class="trace-step">
      <header class="trace-head">
        <span class="trace-index mono">#{{ step.index }}</span>
        <strong>{{ step.action }}</strong>
        <el-tag :type="decisionTypes[step.decision] || 'info'" size="small" effect="light">
          {{ step.decision }}
        </el-tag>
      </header>

      <dl class="trace-details">
        <div>
          <dt>Observation</dt>
          <dd>{{ step.observation || '—' }}</dd>
        </div>
        <div>
          <dt>State Update</dt>
          <dd>{{ step.state_update || '—' }}</dd>
        </div>
        <div>
          <dt>Decision</dt>
          <dd>{{ step.decision || '—' }}</dd>
        </div>
      </dl>
    </article>
  </div>
</template>

<style scoped>
.trace-stream {
  height: min(62vh, 620px);
  overflow-y: auto;
  padding: 4px 4px 4px 0;
  scroll-behavior: smooth;
}

.trace-step {
  position: relative;
  padding: 14px 16px 14px 20px;
  border: 1px solid #e4e7ec;
  border-left: 3px solid #2563eb;
  border-radius: 7px;
  background: #fff;
}

.trace-step + .trace-step {
  margin-top: 10px;
}

.trace-head {
  display: flex;
  align-items: center;
  gap: 10px;
}

.trace-head strong {
  min-width: 0;
  overflow: hidden;
  color: #101828;
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.trace-head .el-tag {
  margin-left: auto;
}

.trace-index {
  color: #98a2b3;
  font-size: 12px;
}

.trace-details {
  display: grid;
  gap: 8px;
  margin: 12px 0 0;
}

.trace-details > div {
  display: grid;
  grid-template-columns: 104px minmax(0, 1fr);
  gap: 10px;
}

.trace-details dt {
  color: #667085;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 12px;
}

.trace-details dd {
  margin: 0;
  color: #344054;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

@media (max-width: 680px) {
  .trace-details > div {
    grid-template-columns: 1fr;
    gap: 2px;
  }
}
</style>
