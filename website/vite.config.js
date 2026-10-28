import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // Load all env vars (not only VITE_-prefixed) so NGROK_HOST works from .env too
  const env = { ...loadEnv(mode, process.cwd(), ''), ...process.env }
  const tunnelHost = env.VITE_ALLOWED_HOST || env.NGROK_HOST

  const server = {
    host: true,
    port: 5173,
    strictPort: true,
  }

  // Only needed when exposing the dev server through a tunnel (e.g. ngrok over HTTPS)
  if (tunnelHost) {
    server.allowedHosts = [tunnelHost]
    server.hmr = { clientPort: 443 }
  }

  return {
    plugins: [react()],
    server,
    test: {
      environment: 'node',
      include: ['src/**/*.test.{js,jsx}'],
    },
  }
})
