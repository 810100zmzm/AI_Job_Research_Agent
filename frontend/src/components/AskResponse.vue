<script setup>
import { computed } from 'vue'
import { Promotion } from '@element-plus/icons-vue'

const props = defineProps({
  question: { type: String, default: '' },
  questionReason: { type: String, default: '' },
  submitting: { type: Boolean, default: false },
})

const emit = defineEmits(['submit'])
const answer = defineModel({ type: String, default: '' })
const canSubmit = computed(() => Boolean(answer.value.trim()) && !props.submitting)

function submit() {
  if (canSubmit.value) emit('submit')
}
</script>

<template>
  <section class="surface ask-panel" aria-live="polite">
    <div class="ask-copy">
      <span class="ask-label">需要补充一条信息</span>
      <h2>{{ question }}</h2>
      <p v-if="questionReason" class="ask-reason">
        <span>为何需要</span>
        {{ questionReason }}
      </p>
    </div>

    <el-input
      v-model="answer"
      type="textarea"
      :rows="4"
      maxlength="2000"
      show-word-limit
      resize="vertical"
      placeholder="直接回答上面的问题"
      :disabled="submitting"
      @keydown.ctrl.enter.prevent="submit"
      @keydown.meta.enter.prevent="submit"
    />

    <div class="ask-actions">
      <el-button type="primary" :icon="Promotion" :loading="submitting" :disabled="!canSubmit" @click="submit">
        提交并继续
      </el-button>
    </div>
  </section>
</template>

<style scoped>
.ask-panel {
  display: grid;
  gap: 14px;
  margin-bottom: 16px;
  padding: 18px;
  border-left: 4px solid #d97706;
}

.ask-copy {
  display: grid;
  gap: 7px;
}

.ask-label {
  color: #b45309;
  font-size: 12px;
  font-weight: 700;
}

.ask-copy h2 {
  margin: 0;
  color: #101828;
  font-size: 18px;
  line-height: 1.45;
}

.ask-reason {
  margin: 0;
  color: #667085;
  line-height: 1.6;
}

.ask-reason span {
  margin-right: 6px;
  color: #475467;
  font-weight: 600;
}

.ask-actions {
  display: flex;
  justify-content: flex-end;
}

@media (max-width: 680px) {
  .ask-actions .el-button {
    width: 100%;
  }
}
</style>
