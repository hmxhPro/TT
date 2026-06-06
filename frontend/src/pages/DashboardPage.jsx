/**
 * src/pages/DashboardPage.jsx
 * ---------------------------
 * Console home / overview. Replaces the old marketing hero with a useful
 * snapshot: live + cumulative stat cards, a platform-status row, and quick
 * entry cards that route into the three workspaces.
 */

import React, { useEffect, useMemo, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import {
  LayoutDashboard, Activity, History as HistoryIcon, Boxes, Tag,
  Target, GraduationCap,
} from 'lucide-react'
import { getModels, getCategories, getTaskHistory } from '../services/api'
import PageHeader from '../components/PageHeader'

function StatCard({ icon, label, value, tone = 'brand' }) {
  const toneMap = {
    brand:   'bg-brand-50 text-brand-600',
    emerald: 'bg-emerald-50 text-emerald-600',
    sky:     'bg-sky-50 text-sky-600',
    violet:  'bg-violet-50 text-violet-600',
  }
  return (
    <div className="card p-5 flex items-center gap-4">
      <span className={`inline-flex p-3 rounded-xl ${toneMap[tone]}`}>{icon}</span>
      <div className="min-w-0">
        <div className="text-2xl font-bold text-ink-900 tabular-nums">{value}</div>
        <div className="text-xs text-ink-500 mt-0.5">{label}</div>
      </div>
    </div>
  )
}

function QuickCard({ to, icon, iconBg, title, desc, cta, ctaCls }) {
  return (
    <Link to={to} className="card p-5 hover:shadow-soft transition-shadow group">
      <div className="flex items-start gap-3">
        <span className={`p-2.5 rounded-xl ${iconBg}`}>{icon}</span>
        <div className="flex-1">
          <h3 className="font-semibold text-ink-900">{title}</h3>
          <p className="text-ink-500 text-sm mt-1">{desc}</p>
          <p className={`mt-3 text-sm flex items-center gap-1 ${ctaCls}`}>
            {cta} <span className="group-hover:translate-x-0.5 transition-transform">→</span>
          </p>
        </div>
      </div>
    </Link>
  )
}

export default function DashboardPage() {
  const { tasks } = useOutletContext()
  const [counts, setCounts] = useState({ models: null, categories: null, history: null })

  useEffect(() => {
    let alive = true
    Promise.allSettled([getModels(), getCategories(), getTaskHistory({ limit: 200 })]).then(
      ([m, c, h]) => {
        if (!alive) return
        setCounts({
          models: m.status === 'fulfilled' ? m.value.length : null,
          categories: c.status === 'fulfilled' ? c.value.length : null,
          history: h.status === 'fulfilled' ? h.value.length : null,
        })
      }
    )
    return () => { alive = false }
  }, [])

  const activeCount = useMemo(
    () =>
      tasks.filter((t) =>
        ['uploading', 'pending', 'running', 'paused', 'packaging'].includes(t.taskStatus)
      ).length,
    [tasks]
  )

  const fmt = (n) => (n == null ? '—' : n)

  return (
    <>
      <PageHeader
        title="概览"
        subtitle="平台运行状态与快捷入口"
        icon={<LayoutDashboard size={18} />}
        right={
          <span className="hidden md:inline-flex items-center gap-2 text-xs text-emerald-600 bg-emerald-50 border border-emerald-100 px-3 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 soft-pulse" />
            平台运行正常
          </span>
        }
      />

      {/* ── Stat cards ────────────────────────────────────────────────── */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard icon={<Activity size={20} />} label="进行中任务" value={activeCount} tone="brand" />
        <StatCard icon={<HistoryIcon size={20} />} label="历史任务" value={fmt(counts.history)} tone="sky" />
        <StatCard icon={<Boxes size={20} />} label="已训练模型" value={fmt(counts.models)} tone="emerald" />
        <StatCard icon={<Tag size={20} />} label="训练类别" value={fmt(counts.categories)} tone="violet" />
      </div>

      {/* ── Quick entries ─────────────────────────────────────────────── */}
      <h2 className="text-sm font-semibold text-ink-700 mb-3">快捷入口</h2>
      <div className="grid md:grid-cols-3 gap-5">
        <QuickCard
          to="/detect"
          icon={<Target size={18} className="text-white" />}
          iconBg="bg-brand-500"
          title="开始视频检测"
          desc="上传多个视频，输入要检测的目标名词（如：钓鱼台、菜地），实时查看结果。"
          cta="进入工作台"
          ctaCls="text-brand-600"
        />
        <QuickCard
          to="/training"
          icon={<GraduationCap size={18} className="text-white" />}
          iconBg="bg-emerald-500"
          title="训练自定义模型"
          desc="创建类别、上传并框选目标，一键训练属于你自己的检测模型。"
          cta="开始训练"
          ctaCls="text-emerald-600"
        />
        <QuickCard
          to="/history"
          icon={<HistoryIcon size={18} className="text-white" />}
          iconBg="bg-sky-500"
          title="查看历史记录"
          desc="浏览过往检测任务、查看保存的检测帧、下载 ZIP 归档或清理缓存。"
          cta="查看历史"
          ctaCls="text-sky-600"
        />
      </div>
    </>
  )
}
