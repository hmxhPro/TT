/**
 * src/pages/TrainingPage.jsx
 * --------------------------
 * The model-training workspace: left column (category manager + train panel +
 * trained-model list) and a right area that toggles between online annotation
 * and annotated-dataset import. State comes from ConsoleLayout via context.
 */

import React from 'react'
import { useOutletContext } from 'react-router-dom'
import { GraduationCap, MousePointer2, Database } from 'lucide-react'

import CategoryManager from '../components/CategoryManager'
import TrainPanel from '../components/TrainPanel'
import ModelListPanel from '../components/ModelListPanel'
import KonvaAnnotator from '../components/KonvaAnnotator'
import DatasetImporter from '../components/DatasetImporter'
import PageHeader from '../components/PageHeader'
import { TabButton } from '../components/ui'

export default function TrainingPage() {
  const {
    selectedCategory, setSelectedCategory, catReloadToken, modelReloadToken,
    trainTab, setTrainTab, refreshSelectedCategory, handleTrained,
  } = useOutletContext()

  return (
    <>
      <PageHeader
        title="模型训练"
        subtitle="创建类别 · 在线标注或上传已标注数据集 · 一键训练专属检测模型"
        icon={<GraduationCap size={18} />}
      />

      <div className="grid lg:grid-cols-[22rem_1fr] gap-6 items-start">
        {/* ── Left: category / train / models ───────────────────── */}
        <aside className="flex flex-col gap-5">
          <CategoryManager
            selectedId={selectedCategory?.id ?? null}
            onSelect={setSelectedCategory}
            reloadToken={catReloadToken}
          />
          <TrainPanel category={selectedCategory} onTrained={handleTrained} />
          <ModelListPanel reloadToken={modelReloadToken} />
        </aside>

        {/* ── Right: annotate online OR import an annotated dataset ─── */}
        <section className="min-w-0">
          <div className="flex items-center gap-2 mb-4">
            <TabButton
              active={trainTab === 'annotate'}
              onClick={() => setTrainTab('annotate')}
              icon={<MousePointer2 size={15} />}
            >
              在线标注
            </TabButton>
            <TabButton
              active={trainTab === 'import'}
              onClick={() => setTrainTab('import')}
              icon={<Database size={15} />}
            >
              上传数据集
            </TabButton>
          </div>

          {trainTab === 'annotate' ? (
            <KonvaAnnotator category={selectedCategory} onDataChanged={refreshSelectedCategory} />
          ) : (
            <DatasetImporter category={selectedCategory} onDataChanged={refreshSelectedCategory} />
          )}
        </section>
      </div>
    </>
  )
}
