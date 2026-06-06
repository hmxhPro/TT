/**
 * src/layout/Sidebar.jsx
 * ----------------------
 * Light, theme-aware sidebar that shares the content canvas color for a unified
 * look. Top: a brand card with the two institutional logos (kept on pure white
 * so the white-bg JPG blends in) + the product wordmark. Middle: primary nav.
 * Bottom: a dark/light theme toggle, a platform-status chip, and an API link.
 */

import React from 'react'
import { LayoutDashboard, Film, GraduationCap, History, Sun, Moon } from 'lucide-react'
import NavItem from './NavItem'
import { useTheme } from '../hooks/useTheme'

export default function Sidebar() {
  const { theme, toggle } = useTheme()
  const dark = theme === 'dark'

  return (
    <aside className="w-64 shrink-0 sticky top-0 h-screen bg-ink-50 flex flex-col border-r border-ink-200">
      {/* ── Brand ─────────────────────────────────────────────────────── */}
      <div className="p-3 border-b border-ink-200">
        <div className="rounded-xl bg-surface border border-ink-200 shadow-soft overflow-hidden">
          {/* logos kept on pure white — the 铁塔 JPG has a white background */}
          <div className="flex items-center justify-center gap-3 px-3 py-2.5 bg-white">
            <img src="/科大logo.png" alt="中国科学技术大学" className="h-7 object-contain" />
            <span className="w-px h-6 bg-ink-200" />
            <img src="/铁塔logo.jpg" alt="中国铁塔" className="h-7 object-contain" />
          </div>
          <div className="flex items-center gap-2 px-3 py-2 border-t border-ink-200">
            <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-brand-500 text-white font-bold text-xs shadow-brand">
              V
            </span>
            <span className="font-semibold text-ink-900 text-sm tracking-tight">视频检测控制台</span>
          </div>
        </div>
      </div>

      {/* ── Primary nav ───────────────────────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto p-3 flex flex-col gap-1">
        <p className="px-3 pt-2 pb-1 text-[11px] font-medium uppercase tracking-wider text-ink-400">
          导航
        </p>
        <NavItem to="/" end icon={<LayoutDashboard size={17} />}>概览</NavItem>
        <NavItem to="/detect" icon={<Film size={17} />}>视频检测</NavItem>
        <NavItem to="/training" icon={<GraduationCap size={17} />}>模型训练</NavItem>
        <NavItem to="/history" icon={<History size={17} />}>历史记录</NavItem>
      </nav>

      {/* ── Footer ────────────────────────────────────────────────────── */}
      <div className="p-3 border-t border-ink-200 flex flex-col gap-1.5">
        <button
          type="button"
          onClick={toggle}
          title={dark ? '切换到浅色模式' : '切换到深色模式'}
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-ink-600 hover:bg-ink-100 hover:text-ink-900 transition-colors"
        >
          {dark ? <Sun size={16} /> : <Moon size={16} />}
          {dark ? '浅色模式' : '深色模式'}
        </button>
        <span className="inline-flex items-center gap-2 text-xs text-emerald-600 px-3 py-1.5 rounded-lg bg-emerald-50">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 soft-pulse" />
          平台运行正常
        </span>
        <a
          href="/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-ink-400 hover:text-ink-700 px-3 py-1 transition-colors"
        >
          API 文档 →
        </a>
      </div>
    </aside>
  )
}
