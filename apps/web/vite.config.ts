import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8008',
      '/healthz': 'http://127.0.0.1:8008',
      '/livez': 'http://127.0.0.1:8008',
      '/readyz': 'http://127.0.0.1:8008',
      '/v1': 'http://127.0.0.1:8008',
    },
  },
})
