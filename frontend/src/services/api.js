/**
 * src/services/api.js
 * -------------------
 * Axios-based API client for the FastAPI backend.
 */

import axios from 'axios'

// Base URL: empty string means "same host" (works with Vite proxy in dev,
// and when frontend is served by the backend in production).
const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  // No timeout by default — individual calls (upload) may take a long time
  // for large files. Non-upload calls are fast and SSE uses EventSource.
  timeout: 0,
  maxContentLength: Infinity,
  maxBodyLength: Infinity,
})

// ── Request / Response interceptors ──────────────────────────────────────────

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.message ||
      'An unknown error occurred.'
    return Promise.reject(new Error(message))
  }
)

// ── API methods ───────────────────────────────────────────────────────────────

/**
 * Upload a video file.
 * @param {File} file
 * @param {(progressPercent: number) => void} [onProgress]
 * @returns {Promise<UploadResponse>}
 */
export async function uploadVideo(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)

  const { data } = await api.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded * 100) / evt.total))
      }
    },
  })
  return data
}

/**
 * Start a detection task.
 * @param {{ video_id: string, video_filename?: string, prompt: string, detection_interval?: number, enable_vlm?: boolean }} params
 * @returns {Promise<DetectResponse>}
 */
export async function startDetection(params) {
  const { data } = await api.post('/api/detect', params)
  return data
}

/**
 * Get task state (polling fallback).
 * @param {string} taskId
 * @returns {Promise<TaskState>}
 */
export async function getTask(taskId) {
  const { data } = await api.get(`/api/task/${taskId}`)
  return data
}

/**
 * Request cancellation of a running detection task.
 * @param {string} taskId
 */
export async function cancelDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/cancel`)
  return data
}

/**
 * Pause a running detection task.
 * @param {string} taskId
 */
export async function pauseDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/pause`)
  return data
}

/**
 * Resume a paused detection task.
 * @param {string} taskId
 */
export async function resumeDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/resume`)
  return data
}

/**
 * Manually terminate a running task and package results.
 * @param {string} taskId
 */
export async function terminateDetection(taskId) {
  const { data } = await api.post(`/api/task/${taskId}/terminate`)
  return data
}

/**
 * Get the SSE stream URL for a task.
 * @param {string} taskId
 * @returns {string}
 */
export function getStreamUrl(taskId) {
  return `${BASE_URL}/api/stream/${taskId}`
}

/**
 * Get the URL for a single annotated frame image.
 * @param {string} taskId
 * @param {string} filename
 * @returns {string}
 */
export function getFrameUrl(taskId, filename) {
  return `${BASE_URL}/api/frame/${taskId}/${filename}`
}

/**
 * Get the download URL for a finished task's ZIP.
 * @param {string} taskId
 * @returns {string}
 */
export function getDownloadUrl(taskId) {
  return `${BASE_URL}/api/download/${taskId}`
}

// ── History (DB-backed) ─────────────────────────────────────────────────────

/**
 * List past detection tasks, newest first.
 * @param {{ limit?: number, offset?: number, date?: string }} [opts]
 * @returns {Promise<Array<TaskHistoryItem>>}
 */
export async function getTaskHistory({ limit = 100, offset = 0, date } = {}) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (date) params.set('date', date)
  const { data } = await api.get(`/api/tasks?${params.toString()}`)
  return data
}

/**
 * List saved annotated frame filenames for a past task.
 * @param {string} taskId
 * @returns {Promise<string[]>}
 */
export async function getTaskFrames(taskId) {
  const { data } = await api.get(`/api/task/${taskId}/frames`)
  return data
}

/**
 * Hard-delete one history task: DB row + frame dir + ZIP + uploaded video
 * (the upload is kept if other tasks still reference its video_id).
 * Backend returns 409 if the task is still running — caller should cancel
 * it first.
 * @param {string} taskId
 */
export async function deleteHistoryTask(taskId) {
  await api.delete(`/api/task/${taskId}`)
}

/**
 * Wipe ALL history: every DB row, every result dir, every upload.
 * 409 if any task is still active in memory.
 */
export async function deleteAllHistory() {
  await api.delete(`/api/tasks`)
}
