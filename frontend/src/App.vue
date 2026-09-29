<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { DataAnalysis, DocumentAdd, Tickets, UploadFilled } from '@element-plus/icons-vue'

const route = useRoute()
const navigation = [
  { to: '/', label: '分析输入', icon: UploadFilled },
  { to: '/resume', label: '生成简历', icon: DocumentAdd },
  { to: '/history', label: '投递记录', icon: Tickets },
]

const activePath = computed(() => {
  if (route.path.startsWith('/history')) return '/history'
  if (route.path.startsWith('/resume')) return '/resume'
  return '/'
})
const isFullscreenPage = computed(() => route.path.startsWith('/history'))
</script>

<template>
  <div class="app-shell" :class="{ 'app-shell--fullscreen': isFullscreenPage }">
    <header class="app-header">
      <div class="brand-block">
        <span class="brand-mark" aria-hidden="true">
          <el-icon><DataAnalysis /></el-icon>
        </span>
        <div>
          <strong>求职尽调助手</strong>
          <small>JD 证据核对与投递跟踪</small>
        </div>
      </div>

      <nav class="main-nav" aria-label="主导航">
        <RouterLink
          v-for="item in navigation"
          :key="item.to"
          :to="item.to"
          class="nav-link"
          :class="{ active: activePath === item.to }"
        >
          <el-icon><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>
    </header>

    <main class="app-main">
      <RouterView />
    </main>
  </div>
</template>
