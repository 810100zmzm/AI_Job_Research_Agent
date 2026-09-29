import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const apiTarget = env.VITE_PROXY_TARGET || 'http://localhost:8000'
  const proxyPaths = ['/api', '/docs', '/openapi.json', '/redoc', '/health']

  return {
    plugins: [vue()],
    server: {
      host: '127.0.0.1',
      port: 5173,
      proxy: Object.fromEntries(
        proxyPaths.map((path) => [
          path,
          {
            target: apiTarget,
            changeOrigin: true,
          },
        ]),
      ),
    },
  }
})
