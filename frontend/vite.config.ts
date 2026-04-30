import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // REST endpoints (file upload, project settings)
      '/project': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Socket.IO — must proxy the full /ws path
      '/ws': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
