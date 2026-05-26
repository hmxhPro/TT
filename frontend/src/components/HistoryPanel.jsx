/**
 * src/components/HistoryPanel.jsx
 * --------------------------------
 * Floating "past tasks" widget pinned to the bottom-left corner.
 *
 * Collapsed:  small round button with a count badge.
 * Expanded:   ~360px wide panel listing tasks grouped by day
 *             (今天 / 昨天 / YYYY-MM-DD).
 *
 * Clicking a row opens HistoryDetailModal.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { History, X, RefreshCw, Inbox, Trash2 } from 'lucide-react'
import { getTaskHistory, deleteHistoryTask, deleteAllHistory } from '../services/api'
import HistoryDetailModal from './HistoryDetailModal'

const STATUS_META = {
  pending:          { label: '排队中',  cls: 'bg-brand-50 text-brand-700' },
  running:          { label: '检测中',  cls: 'bg-brand-100 text-brand-700' },
  paused:           { label: '已暂停',  cls: 'bg-amber-50 text-amber-700' },
  packaging:        { label: '打包中',  cls: 'bg-brand-100 text-brand-700' },
  finished:         { label: '已完成',  cls: 'bg-emerald-50 text-emerald-700' },
  failed:           { label: '失败',    cls: 'bg-red-50 text-red-700' },
  cancelled:        { label: '已取消',  cls: 'bg-ink-100 text-ink-600' },
  early_terminated: { label: '提前终止', cls: 'bg-amber-50 text-amber-800' },
}

function dayBucket(iso) {
  const d = new Date(iso)
  const today = new Date()
  const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime()
  const diffDays = Math.round((startOf(today) - startOf(d)) / 86_400_000)
  if (diffDays === 0) return '今天'
  if (diffDays === 1) return '昨天'
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function formatTime(iso) {
  const d = new Date(iso)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

export default function HistoryPanel() {
  const [open, setOpen] = useState(false)
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedTaskId, setSelectedTaskId] = useState(null)
  // Track per-row delete in flight, and a separate flag for "wipe all".
  const [deletingId, setDeletingId] = useState(null)
  const [clearingAll, setClearingAll] = useState(false)

  const fetchHistory = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const rows = await getTaskHistory({ limit: 200 })
      setTasks(rows)
    } catch (err) {
      setError(err?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }, [])

  // Initial load on mount so the badge count is correct.
  useEffect(() => {
    fetchHistory()
  }, [fetchHistory])

  // Refresh whenever the user opens the panel (cheap call).
  useEffect(() => {
    if (open) fetchHistory()
  }, [open, fetchHistory])

  const grouped = useMemo(() => {
    const out = new Map()
    for (const t of tasks) {
      const key = dayBucket(t.created_at)
      if (!out.has(key)) out.set(key, [])
      out.get(key).push(t)
    }
    return Array.from(out.entries())
  }, [tasks])

  // ── Delete one row ──────────────────────────────────────────────────
  const handleDelete = useCallback(async (task) => {
    const label = task.video_filename || task.task_id.slice(0, 8)
    const ok = window.confirm(
      `确定删除任务「${label}」吗？\n\n` +
      `此操作不可逆，将同时删除：\n` +
      `  · 该任务的历史记录（数据库）\n` +
      `  · 已保存的检测帧与 ZIP 归档\n` +
      `  · 上传的原始视频（若无其他任务引用）`
    )
    if (!ok) return
    setDeletingId(task.task_id)
    setError(null)
    try {
      await deleteHistoryTask(task.task_id)
      // Optimistic local removal so the UI updates instantly; refetch
      // afterwards to stay in sync with the DB.
      setTasks((prev) => prev.filter((t) => t.task_id !== task.task_id))
      fetchHistory()
    } catch (err) {
      setError(err?.message || '删除失败')
    } finally {
      setDeletingId(null)
    }
  }, [fetchHistory])

  // ── Wipe everything ─────────────────────────────────────────────────
  const handleDeleteAll = useCallback(async () => {
    if (tasks.length === 0) return
    const ok = window.confirm(
      `确定清空全部 ${tasks.length} 条历史任务吗？\n\n` +
      `此操作不可逆，将同时删除：\n` +
      `  · 全部历史记录（数据库）\n` +
      `  · 所有任务的检测帧与 ZIP 归档\n` +
      `  · 所有上传的原始视频文件`
    )
    if (!ok) return
    setClearingAll(true)
    setError(null)
    try {
      await deleteAllHistory()
      setTasks([])
      fetchHistory()
    } catch (err) {
      setError(err?.message || '清空失败')
    } finally {
      setClearingAll(false)
    }
  }, [tasks.length, fetchHistory])

  return (
    <>
      {/* ── Floating launcher (always rendered) ──────────────────────── */}
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          title="过往任务"
          className="fixed bottom-4 left-4 z-30 inline-flex items-center gap-2 px-3.5 py-2.5 rounded-full bg-white border border-ink-200 shadow-lg shadow-ink-900/5 hover:bg-ink-50 text-ink-700 text-sm font-medium transition-colors"
        >
          <History size={16} className="text-brand-500" />
          <span>过往任务</span>
          {tasks.length > 0 && (
            <span className="ml-1 inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1.5 rounded-full bg-brand-500 text-white text-[10px] font-semibold">
              {tasks.length}
            </span>
          )}
        </button>
      )}

      {/* ── Expanded panel ─────────────────────────────────────────── */}
      {open && (
        <div className="fixed bottom-4 left-4 z-30 w-[360px] max-h-[70vh] flex flex-col bg-white border border-ink-200 rounded-2xl shadow-2xl shadow-ink-900/10 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-ink-100">
            <div className="flex items-center gap-2">
              <History size={16} className="text-brand-500" />
              <span className="font-semibold text-ink-800">过往任务</span>
              <span className="text-xs text-ink-500">({tasks.length})</span>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={fetchHistory}
                title="刷新"
                disabled={loading}
                className="p-1.5 rounded-lg text-ink-500 hover:text-brand-600 hover:bg-ink-50 disabled:opacity-50"
              >
                <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              </button>
              <button
                type="button"
                onClick={handleDeleteAll}
                title="清空全部历史"
                disabled={clearingAll || tasks.length === 0}
                className="p-1.5 rounded-lg text-ink-500 hover:text-red-600 hover:bg-red-50 disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-ink-500"
              >
                <Trash2 size={14} className={clearingAll ? 'animate-pulse' : ''} />
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                title="关闭"
                className="p-1.5 rounded-lg text-ink-500 hover:text-red-500 hover:bg-red-50"
              >
                <X size={14} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto">
            {error && (
              <div className="m-3 p-3 rounded-lg border border-red-200 bg-red-50 text-red-700 text-xs">
                ⚠ {error}
              </div>
            )}
            {!error && !loading && tasks.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-ink-400">
                <Inbox size={32} className="mb-2" />
                <span className="text-sm">暂无历史任务</span>
              </div>
            )}
            {grouped.map(([day, items]) => (
              <div key={day}>
                <div className="sticky top-0 px-4 py-1.5 bg-ink-50/95 backdrop-blur text-[11px] font-medium text-ink-500 uppercase tracking-wider">
                  {day}
                </div>
                <ul>
                  {items.map((t) => {
                    const meta = STATUS_META[t.status] ?? { label: t.status, cls: 'bg-ink-100 text-ink-600' }
                    const isDeleting = deletingId === t.task_id
                    return (
                      <li key={t.task_id} className="group relative border-b border-ink-100">
                        <div
                          role="button"
                          tabIndex={0}
                          onClick={() => !isDeleting && setSelectedTaskId(t.task_id)}
                          onKeyDown={(e) => {
                            if (isDeleting) return
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              setSelectedTaskId(t.task_id)
                            }
                          }}
                          className={[
                            'w-full text-left px-4 py-3 pr-10 transition-colors cursor-pointer',
                            isDeleting ? 'opacity-50 pointer-events-none' : 'hover:bg-ink-50',
                          ].join(' ')}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            <span className="flex-1 truncate font-medium text-ink-800 text-sm">
                              {t.video_filename || t.task_id.slice(0, 8)}
                            </span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${meta.cls}`}>
                              {meta.label}
                            </span>
                          </div>
                          <div className="text-xs text-ink-500 truncate">
                            {t.prompt}
                          </div>
                          <div className="mt-1 flex items-center gap-3 text-[11px] text-ink-400">
                            <span>{formatTime(t.created_at)}</span>
                            {t.processed_frames > 0 && (
                              <span>{t.processed_frames}/{t.total_frames || '?'} 帧</span>
                            )}
                            {t.zip_ready && <span className="text-emerald-600">ZIP 可下载</span>}
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDelete(t)
                          }}
                          disabled={isDeleting}
                          title="删除该任务及所有缓存"
                          className="absolute top-2 right-2 p-1.5 rounded-lg text-ink-400 hover:text-red-600 hover:bg-red-50 transition-colors disabled:cursor-wait"
                        >
                          <Trash2 size={13} className={isDeleting ? 'animate-pulse' : ''} />
                        </button>
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Detail modal ────────────────────────────────────────────── */}
      {selectedTaskId && (
        <HistoryDetailModal
          taskId={selectedTaskId}
          onClose={() => setSelectedTaskId(null)}
        />
      )}
    </>
  )
}
