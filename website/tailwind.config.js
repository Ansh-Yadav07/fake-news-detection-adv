import animate from 'tailwindcss-animate'

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        background: '#f5f5f6',
      },
      // Layered soft shadows for the results view (hairline + ambient + key)
      boxShadow: {
        card: '0 1px 2px rgba(24, 24, 27, 0.04), 0 2px 6px -2px rgba(24, 24, 27, 0.04), 0 12px 32px -16px rgba(24, 24, 27, 0.10)',
        hero: '0 1px 2px rgba(24, 24, 27, 0.04), 0 4px 12px -4px rgba(24, 24, 27, 0.06), 0 32px 64px -32px rgba(24, 24, 27, 0.18)',
      },
    },
  },
  plugins: [animate],
}