/**
 * src/layout/ConsoleLayout.jsx
 * ----------------------------
 * The application shell AND the single host for cross-page state.
 *
 * It stays mounted for the whole session, so detection SSE streams and
 * uploaded-task state survive navigation between pages (react-router only swaps
 * the <Outlet> child, never this layout). All detection/training state lives
 * here and is handed to pages via <Outlet context={...}/>; pages read it with
 * useOutletContext().
 */

import React, { useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import { useDetectionTasks } from '../hooks/useDetectionTasks'
import { getCategory } from '../services/api'

export default function ConsoleLayout() {
  // ── Detection workspace config ─────────────────────────────────────────
  const [prompt, setPrompt] = useState('')
  const [detInterval, setDetInterval] = useState(5)
  const [enableVlm, setEnableVlm] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)
  // Trained-model selection for the detection workspace ('' = open-vocab).
  const [selectedModelId, setSelectedModelId] = useState('')

  // ── Training workspace state ───────────────────────────────────────────
  const [selectedCategory, setSelectedCategory] = useState(null)
  const [catReloadToken, setCatReloadToken] = useState(0)
  // Shared across pages: bumped after a training run AND read by
  // DetectModelSelect, so a model trained on /training shows up on /detect.
  const [modelReloadToken, setModelReloadToken] = useState(0)
  // Dataset source for the selected category: online annotation vs. import.
  const [trainTab, setTrainTab] = useState('annotate')

  // Instantiated exactly ONCE here (never unmounts on nav) so in-flight SSE
  // streams keep running while the user is on another page.
  const detection = useDetectionTasks()

  // Selecting a trained model defaults VLM off (the model is self-sufficient);
  // it stays user-toggleable in advanced settings. Clearing restores default.
  const handleSelectModel = (id) => {
    setSelectedModelId(id)
    setEnableVlm(!id)
  }

  const handleStartAll = async () => {
    await detection.startAll(prompt.trim(), detInterval, enableVlm, selectedModelId || undefined)
  }

  const handleRetry = (id) => {
    if (!selectedModelId && !prompt.trim()) {
      alert('请先填写检测目标或选择已训练模型')
      return
    }
    detection.resetOne(id)
    // Give state a tick before restarting.
    setTimeout(
      () => detection.startOne(id, prompt.trim(), detInterval, enableVlm, selectedModelId || undefined),
      30
    )
  }

  // Re-fetch the selected category so TrainPanel sees fresh annotated/image
  // counts, and nudge the category list to refresh its own counts.
  const refreshSelectedCategory = async () => {
    if (!selectedCategory) return
    try {
      setSelectedCategory(await getCategory(selectedCategory.id))
    } catch {
      /* keep stale copy on transient failure */
    }
    setCatReloadToken((t) => t + 1)
  }

  // After a training run finishes: reload the model list + refresh counts.
  const handleTrained = () => {
    setModelReloadToken((t) => t + 1)
    refreshSelectedCategory()
  }

  // Single context object handed down to every routed page. Rebuilt each render
  // (useDetectionTasks returns a fresh object anyway) — page re-render scope is
  // the same as the old single-component App.
  const ctx = {
    // detection — state + hook actions
    ...detection,
    prompt, setPrompt, detInterval, setDetInterval, enableVlm, setEnableVlm,
    showAdvanced, setShowAdvanced, selectedModelId, setSelectedModelId,
    handleSelectModel, handleStartAll, handleRetry,
    // training — state + actions
    selectedCategory, setSelectedCategory, catReloadToken, setCatReloadToken,
    modelReloadToken, setModelReloadToken, trainTab, setTrainTab,
    refreshSelectedCategory, handleTrained,
  }

  return (
    <div className="min-h-screen flex bg-ink-50 text-ink-800">
      <Sidebar />
      <main className="flex-1 min-w-0 h-screen overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-7">
          <Outlet context={ctx} />
        </div>
      </main>
    </div>
  )
}
