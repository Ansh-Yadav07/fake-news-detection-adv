import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,

    allowedHosts: [
      "unconfirmative-unimmanently-xuan.ngrok-free.dev"
    ],

    hmr: {
      clientPort: 443
    }
  }
})