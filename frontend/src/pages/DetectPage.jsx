/**
 * src/pages/DetectPage.jsx
 * ------------------------
 * The video-detection workspace: left config column (upload + model/prompt +
 * advanced settings + start) and a right task grid. All state comes from the
 * ConsoleLayout via useOutletContext so tasks keep streaming across navigation.
 */

import React, { useMemo } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Play, Settings, Trash2, Activity, CircleDot, Film, PackageSearch,
} from 'lucide-react'

import VideoUploader from '../components/VideoUploader'
import PromptInput from '../components/PromptInput'
import DetectModelSelect from '../components/DetectModelSelect'
import TaskCard from '../components/TaskCard'
import PageHeader from '../components/PageHeader'
import { StatRow } from '../components/ui'

export default function DetectPage() {
  const {
    tasks, addFiles, removeTask, clearAll,
    cancel, pause, resume, terminate,
    prompt, setPrompt, detInterval, setDetInterval, enableVlm, setEnableVlm,
    showAdvanced, setShowAdvanced, selectedModelId, modelReloadToken,
    handleSelectModel, handleStartAll, handleRetry,
  } = useOutletContext()

  const counts = useMemo(() => {
    const c = {
      queued: 0, uploading: 0, running: 0, paused: 0, packaging: 0,
      finished: 0, failed: 0, cancelled: 0, pending: 0, early_terminated: 0,
    }
    for (const t of tasks) c[t.taskStatus] = (c[t.taskStatus] || 0) + 1
    return c
  }, [tasks])

  const hasWork = tasks.length > 0
  const anyActive = ['uploading', 'pending', 'running', 'paused', 'packaging'].some(
    (s) => (counts[s] || 0) > 0
  )
  const queuedOrFailed = (counts.queued || 0) + (counts.failed || 0) + (counts.cancelled || 0)
  const canStart = hasWork && (selectedModelId || prompt.trim().length > 0) && queuedOrFailed > 0 && !anyActive

  return (
    <>
      <PageHeader
        title="视频检测"
        subtitle="批量上传视频 · 名词列表描述目标 · 独立任务并发处理"
        icon={<Film size={18} />}
        right={
          <span className="hidden md:inline-flex items-center gap-2 text-xs text-emerald-600 bg-emerald-50 border border-emerald-100 px-3 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 soft-pulse" />
            实时结果流
          </span>
        }
      />

      <div className="grid lg:grid-cols-[22rem_1fr] gap-6">
        {/* ── Left: config ──────────────────────────────────────── */}
        <aside className="flex flex-col gap-5">
          <div className="card p-5 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-ink-900 flex items-center gap-2">
                <span className="p-1.5 rounded-lg bg-brand-50 text-brand-500">
                  <Film size={14} />
                </span>
                上传视频
              </h3>
              {hasWork && (
                <button
                  type="button"
                  onClick={clearAll}
                  disabled={anyActive}
                  className={[
                    'text-xs flex items-center gap-1',
                    anyActive ? 'text-ink-300 cursor-not-allowed' : 'text-ink-500 hover:text-red-500',
                  ].join(' ')}
                >
                  <Trash2 size={12} />
                  清空全部
                </button>
              )}
            </div>

            <VideoUploader onFilesSelected={addFiles} disabled={false} hasTasks={hasWork} />

            <DetectModelSelect
              value={selectedModelId}
              onChange={handleSelectModel}
              disabled={anyActive}
              reloadToken={modelReloadToken}
            />

            {!selectedModelId && (
              <PromptInput value={prompt} onChange={setPrompt} disabled={anyActive} />
            )}

            {/* Advanced toggle */}
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="flex items-center gap-2 text-ink-500 hover:text-ink-800 text-sm transition-colors"
            >
              <Settings size={14} />
              高级设置
              <span className="ml-auto">{showAdvanced ? '▲' : '▼'}</span>
            </button>

            {showAdvanced && (
              <div className="flex flex-col gap-3 pl-3 border-l-2 border-brand-100">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableVlm}
                    onChange={(e) => setEnableVlm(e.target.checked)}
                    disabled={anyActive}
                    className="accent-brand-500 w-4 h-4"
                  />
                  <div className="flex flex-col">
                    <span className="text-ink-700 text-sm">VLM 语义复核</span>
                    <span className="text-ink-400 text-xs">
                      使用 MiniCPM-V 对检测结果进行语义验证，提升精度
                    </span>
                  </div>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-ink-700 text-sm">检测间隔（每 N 帧全量检测）</span>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      min={1}
                      max={30}
                      value={detInterval}
                      onChange={(e) => setDetInterval(Number(e.target.value))}
                      disabled={anyActive}
                      className="flex-1 accent-brand-500"
                    />
                    <span className="text-brand-600 font-mono w-8 text-center">{detInterval}</span>
                  </div>
                  <span className="text-ink-400 text-xs">值越大速度越快，精度略降。推荐 3 ~ 10</span>
                </label>
              </div>
            )}

            <button
              type="button"
              onClick={handleStartAll}
              disabled={!canStart}
              className={[
                'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold transition-all duration-200',
                canStart
                  ? 'bg-brand-500 text-white hover:bg-brand-600 shadow-brand'
                  : 'bg-ink-100 text-ink-400 cursor-not-allowed',
              ].join(' ')}
            >
              {anyActive ? <Activity size={16} className="animate-pulse" /> : <Play size={16} />}
              {anyActive
                ? `处理中…（${counts.running + counts.uploading + counts.pending + counts.paused + counts.packaging} 个任务）`
                : queuedOrFailed > 0
                ? `开始检测（${queuedOrFailed} 个视频）`
                : hasWork
                ? '全部已处理'
                : '开始检测'}
            </button>
          </div>

          {/* Stats card */}
          {hasWork && (
            <div className="card p-4 text-sm">
              <h4 className="text-ink-500 font-medium mb-3 text-xs uppercase tracking-wider">任务统计</h4>
              <div className="grid grid-cols-2 gap-y-2 gap-x-4">
                <StatRow label="视频总数" value={tasks.length} />
                <StatRow label="等待中" value={counts.queued || 0} tone="ink" />
                <StatRow
                  label="进行中"
                  value={(counts.uploading || 0) + (counts.pending || 0) + (counts.running || 0) + (counts.packaging || 0)}
                  tone="brand"
                />
                <StatRow label="已完成" value={counts.finished || 0} tone="emerald" />
                {counts.failed > 0 && <StatRow label="失败" value={counts.failed} tone="red" />}
              </div>
            </div>
          )}
        </aside>

        {/* ── Right: task grid ──────────────────────────────────── */}
        <section className="min-w-0">
          {!hasWork ? (
            <div className="card p-12 flex flex-col items-center justify-center text-center gap-3 h-full">
              <span className="p-4 rounded-2xl bg-brand-50 text-brand-500">
                <PackageSearch size={32} />
              </span>
              <h3 className="text-lg font-semibold text-ink-800">还没有上传视频</h3>
              <p className="text-ink-500 text-sm max-w-sm">
                在左侧上传区拖拽或选择视频，再填写检测目标，即可开始批量处理。
              </p>
              <div className="mt-2 flex flex-wrap justify-center gap-2 text-xs">
                <span className="chip"><CircleDot size={10} className="text-brand-500" /> 多视频并发</span>
                <span className="chip"><CircleDot size={10} className="text-brand-500" /> 不限文件大小</span>
                <span className="chip"><CircleDot size={10} className="text-brand-500" /> 实时结果流</span>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4">
              {tasks.map((t) => (
                <TaskCard
                  key={t.id}
                  task={t}
                  onRemove={removeTask}
                  onRetry={handleRetry}
                  onCancel={cancel}
                  onPause={pause}
                  onResume={resume}
                  onTerminate={terminate}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </>
  )
}
