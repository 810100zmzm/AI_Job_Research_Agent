import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'input',
    component: () => import('../views/InputView.vue'),
  },
  {
    path: '/progress/:taskId',
    name: 'progress',
    component: () => import('../views/ProgressView.vue'),
  },
  {
    path: '/report/:taskId',
    name: 'report',
    component: () => import('../views/ReportView.vue'),
  },
  {
    path: '/resume',
    name: 'resume',
    component: () => import('../views/ResumeBuilderView.vue'),
  },
  {
    path: '/history',
    name: 'history',
    component: () => import('../views/HistoryView.vue'),
  },
]

export default createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})
