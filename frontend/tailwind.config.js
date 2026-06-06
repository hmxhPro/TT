/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#fff7ed',
          100: '#ffedd5',
          200: '#fed7aa',
          300: '#fdba74',
          400: '#fb923c',
          500: '#f97316',
          600: '#ea580c',
          700: '#c2410c',
          800: '#9a3412',
          900: '#7c2d12',
        },
        // Neutral scale is driven by CSS variables (see index.css) so it flips
        // between light and dark under the `.dark` class. In light mode the
        // values are identical to the original slate palette.
        ink: {
          50:  'rgb(var(--ink-50) / <alpha-value>)',
          100: 'rgb(var(--ink-100) / <alpha-value>)',
          200: 'rgb(var(--ink-200) / <alpha-value>)',
          300: 'rgb(var(--ink-300) / <alpha-value>)',
          400: 'rgb(var(--ink-400) / <alpha-value>)',
          500: 'rgb(var(--ink-500) / <alpha-value>)',
          600: 'rgb(var(--ink-600) / <alpha-value>)',
          700: 'rgb(var(--ink-700) / <alpha-value>)',
          800: 'rgb(var(--ink-800) / <alpha-value>)',
          900: 'rgb(var(--ink-900) / <alpha-value>)',
        },
        // Elevated surface (cards, inputs, panels) — white in light, dark in dark.
        surface: 'rgb(var(--surface) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['"Inter"', '"PingFang SC"', '"Microsoft YaHei"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'soft':  '0 4px 20px -6px rgba(15, 23, 42, 0.08)',
        'brand': '0 10px 30px -10px rgba(249, 115, 22, 0.45)',
      },
      backgroundImage: {
        'hero-glow':
          'radial-gradient(900px 500px at 50% -10%, rgba(249,115,22,0.10), transparent 60%), radial-gradient(700px 400px at 15% 30%, rgba(253,186,116,0.18), transparent 60%)',
      },
    },
  },
  plugins: [],
}
