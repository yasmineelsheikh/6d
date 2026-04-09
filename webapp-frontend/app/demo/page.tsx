'use client'

import { useState } from 'react'
import {
  Bot, Upload, Sparkles, ShieldCheck, Tag, Package,
  Check, Eye, HardDrive, FolderOpen, X,
} from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────

type DataSource = 'collect' | 'upload'
type DataType = 'Teleoperation' | 'Motion Capture' | 'Egocentric'
type UploadMethod = 'connect' | 'files'
type StorageProvider = 'AWS S3'
type QualityMode = 'check-only' | 'check-and-fix'
type AugGoal = 'More diversity' | 'New objects' | 'New environments' | 'More volume'
type AnnotationType = '3D scene reconstruction' | 'Hand pose tracking' | 'Point tracking' | 'Object 6DoF pose'

interface Stage {
  id: string
  name: string
  description: string
  icon: React.ComponentType<{ className?: string }>
  status: 'active' | 'pending'
}

// ─── Shared UI Primitives ─────────────────────────────────────────────────────

function SegmentedControl<T extends string>({
  options, value, onChange,
}: { options: T[]; value: T | null; onChange: (v: T) => void }) {
  return (
    <div className="flex flex-wrap gap-1 bg-gray-100 rounded-lg p-1">
      {options.map(opt => (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${value === opt
            ? 'bg-white text-gray-900 shadow-sm'
            : 'text-gray-500 hover:text-gray-700'
            }`}
        >
          {opt}
        </button>
      ))}
    </div>
  )
}

function Chips<T extends string>({
  options, selected, onToggle,
}: { options: T[]; selected: T[]; onToggle: (v: T) => void }) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map(opt => {
        const active = selected.includes(opt)
        return (
          <button
            key={opt}
            onClick={() => onToggle(opt)}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-sm border transition-all ${active
              ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
              : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
              }`}
          >
            {active && <Check className="w-3 h-3" />}
            {opt}
          </button>
        )
      })}
    </div>
  )
}


function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className={`relative w-10 h-6 rounded-full transition-colors focus:outline-none flex-shrink-0 ${enabled ? 'bg-indigo-600' : 'bg-gray-200'
        }`}
    >
      <span
        className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-sm transition-transform ${enabled ? 'translate-x-4' : ''
          }`}
      />
    </button>
  )
}

// ─── Section Wrappers ─────────────────────────────────────────────────────────

function SectionCard({
  number, title, done, summary, onEdit, children,
}: {
  number: number
  title: string
  done: boolean
  summary?: string
  onEdit?: () => void
  children: React.ReactNode
}) {
  return (
    <div className={`bg-white rounded-xl border transition-all ${done ? 'border-gray-200' : 'border-gray-200 shadow-sm'
      }`}>
      <div className="px-5 py-4 flex items-center gap-3">
        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold flex-shrink-0 ${done ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-500'
          }`}>
          {done ? <Check className="w-3.5 h-3.5" /> : number}
        </div>
        <span className={`font-medium text-sm ${done ? 'text-gray-500' : 'text-gray-900'}`}>
          {title}
        </span>
        {done && summary && (
          <span className="text-sm text-gray-400 truncate">· {summary}</span>
        )}
        {done && onEdit && (
          <button
            onClick={onEdit}
            className="ml-auto text-xs text-indigo-600 hover:text-indigo-800 font-medium flex-shrink-0"
          >
            Edit
          </button>
        )}
      </div>
      {!done && (
        <div className="px-5 pb-5 border-t border-gray-100 pt-4">
          {children}
        </div>
      )}
    </div>
  )
}

function OptionalSection({
  number, title, description, enabled, onToggle, children,
}: {
  number: number
  title: string
  description?: string
  enabled: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className={`bg-white rounded-xl border transition-all ${enabled ? 'border-indigo-200 shadow-sm' : 'border-gray-200'
      }`}>
      <div className="px-5 py-4 flex items-center gap-3">
        <div className="w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center text-xs font-semibold text-gray-400 flex-shrink-0">
          {number}
        </div>
        <div className="flex-1 min-w-0">
          <span className="font-medium text-sm text-gray-700">{title}</span>
          {description && !enabled && (
            <p className="text-xs text-gray-400 mt-0.5">{description}</p>
          )}
        </div>
        <Toggle enabled={enabled} onToggle={onToggle} />
      </div>
      {enabled && (
        <div className="px-5 pb-5 border-t border-indigo-100 pt-4">
          {children}
        </div>
      )}
    </div>
  )
}

// ─── Pipeline Screen ──────────────────────────────────────────────────────────

function PipelineScreen({ stages, onBack }: { stages: Stage[]; onBack: () => void }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="max-w-2xl mx-auto px-6 py-10">
        <div className="mb-8">
          <button
            onClick={onBack}
            className="text-sm text-indigo-600 hover:text-indigo-800 mb-3 block"
          >
            ← New request
          </button>
          <h1 className="text-2xl font-semibold text-gray-900">Pipeline</h1>
          <p className="text-sm text-gray-500 mt-1">
            Built from your request · {stages.length} stage{stages.length !== 1 ? 's' : ''}
          </p>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          {stages.map((stage, i) => {
            const Icon = stage.icon
            const isLast = i === stages.length - 1
            const isActive = stage.status === 'active'

            return (
              <div
                key={stage.id}
                className={`flex gap-4 px-6 py-5 ${!isLast ? 'border-b border-gray-100' : ''} ${isActive ? 'bg-indigo-50/60' : ''
                  }`}
              >
                {/* Icon + connector */}
                <div className="flex flex-col items-center">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${isActive ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-400'
                    }`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  {!isLast && (
                    <div
                      className={`w-px mt-2 flex-1 ${isActive ? 'bg-indigo-200' : 'bg-gray-100'}`}
                      style={{ minHeight: 16 }}
                    />
                  )}
                </div>

                {/* Content */}
                <div className="flex-1 pt-1.5">
                  <div className="flex items-center gap-2">
                    <span className={`font-medium text-sm ${isActive ? 'text-gray-900' : 'text-gray-500'}`}>
                      {stage.name}
                    </span>
                    {isActive ? (
                      <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-medium">
                        Active
                      </span>
                    ) : (
                      <span className="text-xs text-gray-400">Pending</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">{stage.description}</p>
                </div>
              </div>
            )
          })}
        </div>

        <p className="text-xs text-gray-400 text-center mt-6">
          Pipeline updates in real time as your data moves through each stage.
        </p>
      </main>
    </div>
  )
}

// ─── Header ───────────────────────────────────────────────────────────────────

function Header() {
  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 bg-gray-900 rounded-lg flex items-center justify-center">
          <span className="text-white text-xs font-bold">6d</span>
        </div>
        <span className="text-gray-900 font-semibold text-sm">6d labs</span>
        <span className="text-gray-300 mx-0.5">·</span>
      </div>
    </header>
  )
}

// ─── Main Demo Page ───────────────────────────────────────────────────────────

export default function DemoPage() {
  const [screen, setScreen] = useState<'request' | 'pipeline'>('request')

  // Section 1
  const [dataSource, setDataSource] = useState<DataSource | null>(null)
  const [s1Done, setS1Done] = useState(false)

  // Section 2 — Collect
  const [dataTypes, setDataTypes] = useState<DataType[]>([])
  const [targetHours, setTargetHours] = useState<string>('')
  const [additionalSpecs, setAdditionalSpecs] = useState('')

  // Section 2 — Upload
  const [uploadMethod, setUploadMethod] = useState<UploadMethod | null>(null)
  const [storageProvider, setStorageProvider] = useState<StorageProvider | null>(null)
  const [bucketPath, setBucketPath] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [s2Done, setS2Done] = useState(false)

  // Section 3 — Augmentation
  const [augEnabled, setAugEnabled] = useState(false)
  const [augGoals, setAugGoals] = useState<AugGoal[]>([])
  const [augMultiplier, setAugMultiplier] = useState<string | null>(null)

  // Section 4 — Quality
  const [qualityEnabled, setQualityEnabled] = useState(false)
  const [qualityMode, setQualityMode] = useState<QualityMode>('check-and-fix')

  // Section 5 — Annotation
  const [annotationEnabled, setAnnotationEnabled] = useState(false)
  const [annotations, setAnnotations] = useState<AnnotationType[]>([])

  // UI
  const [showReview, setShowReview] = useState(false)

  // Helpers
  const toggleDataType = (t: DataType) =>
    setDataTypes(p => p.includes(t) ? p.filter(x => x !== t) : [...p, t])
  const toggleAugGoal = (g: AugGoal) =>
    setAugGoals(p => p.includes(g) ? p.filter(x => x !== g) : [...p, g])
  const toggleAnnotation = (a: AnnotationType) =>
    setAnnotations(p => p.includes(a) ? p.filter(x => x !== a) : [...p, a])

  const s2Valid = dataSource === 'collect'
    ? (dataTypes.length > 0 && targetHours !== '')
    : uploadMethod === 'connect'
      ? (storageProvider !== null && bucketPath.trim() !== '')
      : uploadMethod === 'files'
        ? uploadedFiles.length > 0
        : false

  const s1Summary = dataSource === 'collect' ? 'Collect new data' : 'Upload existing data'

  const s2Summary = dataSource === 'collect'
    ? [dataTypes.join(', '), targetHours ? `${targetHours} hrs` : null].filter(Boolean).join(' · ')
    : uploadMethod === 'connect'
      ? [storageProvider, bucketPath].filter(Boolean).join(' · ')
      : `${uploadedFiles.length} file${uploadedFiles.length !== 1 ? 's' : ''}`

  function getPipelineStages(): Stage[] {
    const raw: Omit<Stage, 'status'>[] = []

    if (dataSource === 'collect') {
      raw.push({
        id: 'collection',
        name: 'Collection',
        description: [dataTypes.length > 0 && `${dataTypes.join(', ')}`, targetHours && `${targetHours} hrs`].filter(Boolean).join(' · '),
        icon: Bot,
      })
    } else {
      raw.push({
        id: 'ingestion',
        name: 'Ingestion',
        description: uploadMethod === 'connect'
          ? [storageProvider, bucketPath].filter(Boolean).join(' · ') || 'Cloud storage ingestion'
          : uploadedFiles.length > 0
            ? `${uploadedFiles.length} file${uploadedFiles.length !== 1 ? 's' : ''} uploaded`
            : 'Processing uploaded dataset',
        icon: Upload,
      })
    }

    if (augEnabled) {
      raw.push({
        id: 'augmentation',
        name: 'Augmentation',
        description: [
          augGoals.length > 0 && augGoals.join(', '),
          augMultiplier && `${augMultiplier}× existing volume`,
        ].filter(Boolean).join(' · ') || 'Scaling dataset with synthetic augmentation',
        icon: Sparkles,
      })
    }

    if (qualityEnabled) {
      raw.push({
        id: 'quality',
        name: 'Quality Processing',
        description: qualityMode === 'check-only'
          ? 'Scanning for corruption, sync drift, and spec mismatches'
          : 'Fixing issues, removing unfixable samples',
        icon: ShieldCheck,
      })
      if (qualityMode === 'check-and-fix') {
        raw.push({
          id: 'review-gate',
          name: 'Review Gate',
          description: 'Human approval required before pipeline continues',
          icon: Eye,
        })
      }
    }

    if (annotationEnabled) {
      raw.push({
        id: 'annotation',
        name: 'Annotation',
        description: annotations.length > 0 ? annotations.join(' · ') : 'Labeling and pose estimation',
        icon: Tag,
      })
    }

    raw.push({
      id: 'delivery',
      name: 'Ready for Delivery',
      description: 'Dataset packaged, validated, and ready for training',
      icon: Package,
    })

    return raw.map((s, i) => ({ ...s, status: i === 0 ? 'active' : 'pending' }))
  }

  const reviewItems = [
    { label: 'Source', value: s1Summary },
    ...(dataSource === 'collect' ? [
      { label: 'Data Type', value: dataTypes.length > 0 ? dataTypes.join(', ') : '—' },
      { label: 'Target Hours', value: targetHours ? `${targetHours} hrs` : '—' },
      ...(additionalSpecs ? [{ label: 'Details', value: additionalSpecs }] : []),
    ] : [
      { label: 'Method', value: uploadMethod === 'connect' ? 'Connect Storage' : 'Upload Files' },
      ...(uploadMethod === 'connect' ? [
        { label: 'Provider', value: storageProvider ?? '—' },
        ...(bucketPath ? [{ label: 'Path', value: bucketPath }] : []),
      ] : [
        { label: 'Files', value: `${uploadedFiles.length} file${uploadedFiles.length !== 1 ? 's' : ''}` },
      ]),
    ]),
    ...(augEnabled ? [
      { label: 'Augmentation', value: augGoals.length > 0 ? augGoals.join(', ') : 'Enabled' },
      ...(augMultiplier ? [{ label: 'Scale', value: `${augMultiplier}× existing data` }] : []),
    ] : []),
    ...(qualityEnabled ? [
      { label: 'Quality', value: qualityMode === 'check-only' ? 'Check only' : 'Check and fix' },
    ] : []),
    ...(annotationEnabled ? [
      { label: 'Annotations', value: annotations.length > 0 ? annotations.join(', ') : 'Enabled' },
    ] : []),
  ]

  if (screen === 'pipeline') {
    return <PipelineScreen stages={getPipelineStages()} onBack={() => setScreen('request')} />
  }

  const handleEditS1 = () => {
    setS1Done(false)
    setS2Done(false)
    setShowReview(false)
  }
  const handleEditS2 = () => {
    setS2Done(false)
    setShowReview(false)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="max-w-2xl mx-auto px-6 py-10 space-y-3">
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-gray-900">New Request</h1>
        </div>

        {/* ── Section 1: Data Source ── */}
        <SectionCard
          number={1}
          title="Data Source"
          done={s1Done}
          summary={s1Summary}
          onEdit={handleEditS1}
        >
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {[
                {
                  value: 'collect' as DataSource,
                  icon: Bot,
                  title: 'Collect',
                },
                {
                  value: 'upload' as DataSource,
                  icon: Upload,
                  title: 'Upload',
                },
              ].map(({ value, icon: Icon, title }) => (
                <button
                  key={value}
                  onClick={() => setDataSource(value)}
                  className={`p-4 rounded-xl border-2 text-left transition-all ${dataSource === value
                    ? 'border-indigo-400 bg-indigo-50'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                >
                  <Icon className={`w-5 h-5 mb-2.5 ${dataSource === value ? 'text-indigo-600' : 'text-gray-400'}`} />
                  <div className={`font-semibold text-sm ${dataSource === value ? 'text-indigo-700' : 'text-gray-800'}`}>
                    {title}
                  </div>
                </button>
              ))}
            </div>
            {dataSource && (
              <button
                onClick={() => setS1Done(true)}
                className="w-full py-2.5 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-700 transition-colors"
              >
                Continue →
              </button>
            )}
          </div>
        </SectionCard>

        {/* ── Section 2: Specification ── */}
        {s1Done && (
          <SectionCard
            number={2}
            title="Specification"
            done={s2Done}
            summary={s2Summary}
            onEdit={handleEditS2}
          >
            {dataSource === 'collect' ? (
              <div className="space-y-6">
                <div>
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                    Data Type
                  </label>
                  <Chips<DataType>
                    options={['Teleoperation', 'Motion Capture', 'Egocentric']}
                    selected={dataTypes}
                    onToggle={toggleDataType}
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                    Target Hours
                  </label>
                  <input
                    type="number"
                    min="0"
                    placeholder="Enter total hours"
                    value={targetHours}
                    onChange={e => setTargetHours(e.target.value)}
                    className="w-full bg-white border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:border-indigo-300"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                    Additional Specifications
                  </label>
                  <textarea
                    rows={3}
                    placeholder="Enter any other requirements..."
                    value={additionalSpecs}
                    onChange={e => setAdditionalSpecs(e.target.value)}
                    className="w-full bg-white border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:border-indigo-300"
                  />
                </div>
                {s2Valid && (
                  <button
                    onClick={() => setS2Done(true)}
                    className="w-full py-2.5 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-700 transition-colors"
                  >
                    Continue →
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                {/* Method selector */}
                <div className="grid grid-cols-2 gap-3">
                  {([
                    { value: 'connect' as UploadMethod, icon: HardDrive, label: 'Connect Storage', desc: 'Point to data in your cloud bucket' },
                    { value: 'files' as UploadMethod, icon: FolderOpen, label: 'Upload Files', desc: 'Drag and drop files directly' },
                  ]).map(({ value, icon: Icon, label, desc }) => (
                    <button
                      key={value}
                      onClick={() => setUploadMethod(value)}
                      className={`p-4 rounded-xl border-2 text-left transition-all ${
                        uploadMethod === value
                          ? 'border-indigo-400 bg-indigo-50'
                          : 'border-gray-200 bg-white hover:border-gray-300'
                      }`}
                    >
                      <Icon className={`w-5 h-5 mb-2.5 ${uploadMethod === value ? 'text-indigo-600' : 'text-gray-400'}`} />
                      <div className={`font-semibold text-sm ${uploadMethod === value ? 'text-indigo-700' : 'text-gray-800'}`}>{label}</div>
                      <div className="text-xs text-gray-500 mt-0.5">{desc}</div>
                    </button>
                  ))}
                </div>

                {/* Connect Storage fields */}
                {uploadMethod === 'connect' && (
                  <div className="space-y-4 pt-1">
                    <div>
                      <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                        Storage Provider
                      </label>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setStorageProvider('AWS S3')}
                          className={`px-4 py-2 rounded-lg text-sm border font-medium transition-all ${
                            storageProvider === 'AWS S3'
                              ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                              : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                          }`}
                        >
                          AWS S3
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                        Bucket / Path
                      </label>
                      <input
                        type="text"
                        placeholder="e.g. s3://my-bucket/collections/run-42"
                        value={bucketPath}
                        onChange={e => setBucketPath(e.target.value)}
                        className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-800 bg-white focus:outline-none focus:border-indigo-300 font-mono"
                      />
                    </div>
                  </div>
                )}

                {/* Upload Files drop zone */}
                {uploadMethod === 'files' && (
                  <div className="pt-1">
                    <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                      Files
                    </label>
                    <div
                      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
                      onDragLeave={() => setDragOver(false)}
                      onDrop={e => {
                        e.preventDefault()
                        setDragOver(false)
                        setUploadedFiles(prev => [...prev, ...Array.from(e.dataTransfer.files)])
                      }}
                      onClick={() => document.getElementById('file-input')?.click()}
                      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                        dragOver
                          ? 'border-indigo-400 bg-indigo-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <Upload className={`w-5 h-5 mx-auto mb-2 ${dragOver ? 'text-indigo-400' : 'text-gray-300'}`} />
                      <p className="text-sm text-gray-500">Drop files here or <span className="text-indigo-600">browse</span></p>
                      <input
                        id="file-input"
                        type="file"
                        multiple
                        className="hidden"
                        onChange={e => {
                          if (e.target.files) setUploadedFiles(prev => [...prev, ...Array.from(e.target.files!)])
                        }}
                      />
                    </div>
                    {uploadedFiles.length > 0 && (
                      <ul className="mt-3 space-y-1">
                        {uploadedFiles.map((f, i) => (
                          <li key={i} className="flex items-center gap-2 text-sm text-gray-700 bg-gray-50 rounded-lg px-3 py-2">
                            <span className="flex-1 truncate">{f.name}</span>
                            <button
                              onClick={() => setUploadedFiles(prev => prev.filter((_, j) => j !== i))}
                              className="text-gray-400 hover:text-gray-600"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}

                {s2Valid && (
                  <button
                    onClick={() => setS2Done(true)}
                    className="w-full py-2.5 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-700 transition-colors"
                  >
                    Continue →
                  </button>
                )}
              </div>
            )}
          </SectionCard>
        )}

        {/* ── Sections 3–5: Optional ── */}
        {s1Done && s2Done && (
          <>
            {/* Section 3 — Augmentation */}
            <OptionalSection
              number={3}
              title="Augmentation"
              description="Scale your dataset with synthetic augmentation"
              enabled={augEnabled}
              onToggle={() => setAugEnabled(p => !p)}
            >
              <div className="space-y-5">
                <div>
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                    Goal
                  </label>
                  <Chips<AugGoal>
                    options={['More diversity', 'New objects', 'New environments', 'More volume']}
                    selected={augGoals}
                    onToggle={toggleAugGoal}
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                    Scale Factor
                  </label>
                  <div className="flex gap-2">
                    {['2×', '5×', '10×', '20×'].map(m => (
                      <button
                        key={m}
                        onClick={() => setAugMultiplier(m.replace('×', ''))}
                        className={`flex-1 py-2 px-3 rounded-lg text-sm border font-medium transition-all ${
                          augMultiplier === m.replace('×', '')
                            ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                            : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </OptionalSection>

            {/* Section 4 — Quality */}
            <OptionalSection
              number={4}
              title="Quality Processing"
              description="Catch corruption, sync drift, and spec mismatches"
              enabled={qualityEnabled}
              onToggle={() => setQualityEnabled(p => !p)}
            >
              <div className="space-y-2">
                {(['check-only', 'check-and-fix'] as QualityMode[]).map(mode => (
                  <label
                    key={mode}
                    className={`flex items-start gap-3 p-3.5 rounded-lg border cursor-pointer transition-all ${qualityMode === mode
                      ? 'border-indigo-200 bg-indigo-50'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                      }`}
                  >
                    <input
                      type="radio"
                      name="quality"
                      value={mode}
                      checked={qualityMode === mode}
                      onChange={() => setQualityMode(mode)}
                      className="mt-0.5 accent-indigo-600"
                    />
                    <div>
                      <div className="text-sm font-medium text-gray-800 flex items-center gap-2">
                        {mode === 'check-only' ? 'Check only' : 'Check and fix'}
                        {mode === 'check-and-fix'}
                      </div>
                      <p className="text-xs text-gray-400 mt-0.5">
                        {mode === 'check-only'
                          ? 'Scan and report issues without modifying data'
                          : "Fix what's fixable, remove what isn't"}
                      </p>
                    </div>
                  </label>
                ))}
                <p className="text-xs text-gray-400 pt-1">
                  Catches corruption, sync drift, spec mismatches and diversity gaps
                </p>
              </div>
            </OptionalSection>

            {/* Section 5 — Annotation */}
            <OptionalSection
              number={5}
              title="Annotation"
              description="Label poses, scenes, and tracking targets"
              enabled={annotationEnabled}
              onToggle={() => setAnnotationEnabled(p => !p)}
            >
              <div className="space-y-2">
                {(['3D scene reconstruction', 'Hand pose tracking', 'Point tracking', 'Object 6DoF pose'] as AnnotationType[]).map(ann => (
                  <label
                    key={ann}
                    className={`flex items-center gap-3 p-3.5 rounded-lg border cursor-pointer transition-all ${annotations.includes(ann)
                      ? 'border-indigo-200 bg-indigo-50'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                      }`}
                  >
                    <input
                      type="checkbox"
                      checked={annotations.includes(ann)}
                      onChange={() => toggleAnnotation(ann)}
                      className="accent-indigo-600"
                    />
                    <span className="text-sm text-gray-700">{ann}</span>
                  </label>
                ))}
              </div>
            </OptionalSection>
          </>
        )}

        {/* ── Review & Submit ── */}
        {s1Done && s2Done && !showReview && (
          <button
            onClick={() => setShowReview(true)}
            className="w-full py-3 rounded-xl bg-gray-900 text-white font-medium text-sm hover:bg-gray-700 transition-colors"
          >
            Review Request
          </button>
        )}

        {showReview && (
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Request Summary</h3>
            <div className="space-y-2.5 mb-5">
              {reviewItems.map((item, i) => (
                <div key={i} className="flex gap-4 text-sm">
                  <span className="text-gray-400 w-28 flex-shrink-0">{item.label}</span>
                  <span className="text-gray-800">{item.value}</span>
                </div>
              ))}
            </div>
            <div className="pt-4 border-t border-gray-100 flex gap-3">
              <button
                onClick={() => setShowReview(false)}
                className="flex-1 py-2.5 rounded-lg border border-gray-200 text-sm font-medium text-gray-600 hover:bg-gray-50 transition-colors"
              >
                Edit
              </button>
              <button
                onClick={() => setScreen('pipeline')}
                className="flex-1 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 transition-colors"
              >
                Submit Request →
              </button>
            </div>
          </div>
        )}

        <div className="h-8" />
      </main>
    </div>
  )
}
