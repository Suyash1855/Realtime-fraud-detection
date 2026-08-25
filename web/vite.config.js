import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    // FastAPI mounts this directory as the static root (STATIC_DIR in app/main.py)
    outDir: 'dist',
    emptyOutDir: true,
    // Recharts is the bulk of the bundle; splitting it lets the shell paint first.
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          charts: ['recharts'],
        },
      },
    },
  },
  server: {
    port: 5173,
    // During `npm run dev` the API runs separately on 8000.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
