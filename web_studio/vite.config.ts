import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 6000,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: 'plotly-vendor', test: /plotly\.js/ },
            { name: 'react-vendor', test: /node_modules[\\/](react|react-dom|lucide-react)/ },
          ],
        },
      },
    },
  },
});
