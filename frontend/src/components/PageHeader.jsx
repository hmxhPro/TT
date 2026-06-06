/**
 * src/components/PageHeader.jsx
 * ----------------------------
 * Shared page top-bar for every console page: an optional icon + title +
 * subtitle on the left, and a free-form `right` slot (status chip / actions)
 * on the right.
 */

import React from 'react'

export default function PageHeader({ title, subtitle, icon, right }) {
  return (
    <div className="flex items-start justify-between gap-4 mb-6">
      <div className="flex items-start gap-3 min-w-0">
        {icon && (
          <span className="mt-0.5 p-2 rounded-xl bg-brand-50 text-brand-500 shrink-0">{icon}</span>
        )}
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight text-ink-900">{title}</h1>
          {subtitle && <p className="mt-1 text-sm text-ink-500">{subtitle}</p>}
        </div>
      </div>
      {right && <div className="shrink-0 flex items-center gap-2">{right}</div>}
    </div>
  )
}
