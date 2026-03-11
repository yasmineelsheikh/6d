'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { Loader2, XCircle, Menu, Zap } from 'lucide-react'
import SideMenu from '@/components/SideMenu'
import TaskModal, { TaskData } from '@/components/TaskModal'
import SettingsModal, { SettingsData } from '@/components/SettingsModal'
import BillingModal from '@/components/BillingModal'
import LoginModal from '@/components/LoginModal'
import RegisterModal from '@/components/RegisterModal'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils'

import OverviewTab from '@/components/tabs/OverviewTab'
import HistoryTab from '@/components/tabs/HistoryTab'
import ActionsTab from '@/components/tabs/ActionsTab'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

interface DatasetInfo { dataset_name: string; total_episodes: number; robot_type: string }
interface Visualization { title: string; figure: any }

type TabKey = 'overview' | 'actions' | 'history'
const TABS: { key: TabKey; label: string }[] = [
  { key: 'overview', label: 'Overview' },
  { key: 'actions', label: 'Actions' },
  { key: 'history', label: 'History' },
]

// Seed tasks for SideMenu
const SEED_TASK_NAMES = ['Stack Cups', 'Fold Laundry', 'Open Door', 'Pour Liquid', 'Pick and Place']

export default function DatasetPage() {
  const params = useParams()
  const router = useRouter()
  const searchParams = useSearchParams()
  const datasetName = params?.name as string

  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [datasetData, setDatasetData] = useState<any[]>([])
  const [curatedData, setCuratedData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [distributionsLoading, setDistributionsLoading] = useState(false)
  const [ingestionComplete, setIngestionComplete] = useState(false)
  const [waitingForIngestion, setWaitingForIngestion] = useState(true)
  const [ingestionProgress, setIngestionProgress] = useState(0)
  const [analysisView, setAnalysisView] = useState<'original' | 'new'>('original')
  const [analysisSwitchEnabled, setAnalysisSwitchEnabled] = useState(false)

  const [aresDistributions, setAresDistributions] = useState<Visualization[]>([])
  const [newDistributions, setNewDistributions] = useState<Visualization[]>([])
  const [variationsIncluded, setVariationsIncluded] = useState(false)

  const { user, isAuthenticated, loading: authLoading, logout } = useAuth()
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false)
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false)

  const [environment, setEnvironment] = useState<'Indoor' | 'Outdoor' | ''>('')
  const [selectedAxes, setSelectedAxes] = useState<string[]>([])
  const [isIndoor, setIsIndoor] = useState(false)
  const [isOutdoor, setIsOutdoor] = useState(false)

  const [isSideMenuOpen, setIsSideMenuOpen] = useState(false)
  const [isTaskModalOpen, setIsTaskModalOpen] = useState(false)
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false)
  const [isBillingModalOpen, setIsBillingModalOpen] = useState(false)
  const [tasks, setTasks] = useState<TaskData[]>([])

  const [activeTab, setActiveTab] = useState<TabKey>('overview')

  // Upload state for Actions tab
  const [uploadMode, setUploadMode] = useState<'local' | 's3' | 'huggingface'>('local')
  const [uploadedFiles, setUploadedFiles] = useState<FileList | null>(null)
  const [datasetPath, setDatasetPath] = useState('')
  const [datasetNameInput, setDatasetNameInput] = useState('')
  const [uploadLoading, setUploadLoading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [s3AccessKey, setS3AccessKey] = useState('')
  const [s3SecretKey, setS3SecretKey] = useState('')
  const [s3Bucket, setS3Bucket] = useState('')
  const [s3Region, setS3Region] = useState('')
  const [s3UserPath, setS3UserPath] = useState('')
  const [hfRepoId, setHfRepoId] = useState('')
  const [hfSplit, setHfSplit] = useState('train')
  const [hfToken, setHfToken] = useState('')

  // ── All data-fetching and ingestion logic preserved from original ──

  const handleDatasetLoaded = async (name: string) => {
    setLoading(true); setError(null)
    try {
      const response = await fetch(`${API_BASE}/api/datasets/${name}/info`)
      if (!response.ok) throw new Error('Failed to load dataset info')
      setDatasetInfo(await response.json())
      const dataResponse = await fetch(`${API_BASE}/api/datasets/${name}/data`)
      if (!dataResponse.ok) throw new Error('Failed to load dataset data')
      const data = await dataResponse.json()
      setDatasetData(data.data || [])
    } catch (err: any) { setError(err.message) } finally { setLoading(false) }
  }

  const handleAugmentationComplete = async (dn: string) => {
    try {
      const dataResponse = await fetch(`${API_BASE}/api/datasets/${dn}/data`)
      if (!dataResponse.ok) throw new Error('Failed to load curated data')
      const data = await dataResponse.json()
      setCuratedData(data.data || [])
      setAnalysisSwitchEnabled(true)
    } catch (err: any) { setError(err.message) }
  }

  const handleExportDataset = async () => {
    if (!datasetName) return
    try {
      const response = await fetch(`${API_BASE}/api/datasets/${datasetName}/export`)
      if (!response.ok) throw new Error(await response.text() || 'Failed to export')
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a'); a.href = url; a.download = `${datasetName}_export.zip`
      document.body.appendChild(a); a.click(); document.body.removeChild(a); window.URL.revokeObjectURL(url)
    } catch (err: any) { setError(err.message) }
  }

  const handleSaveTask = async (task: TaskData) => {
    try {
      const response = await fetch(`${API_BASE}/api/tasks`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(task) })
      if (!response.ok) throw new Error('Failed to save task')
      setTasks([...tasks, await response.json()]); setIsTaskModalOpen(false)
    } catch (err: any) { setError(err.message) }
  }

  const handleSaveSettings = async (settings: SettingsData) => {
    try {
      const response = await fetch(`${API_BASE}/api/settings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) })
      if (!response.ok) throw new Error('Failed to save settings'); setIsSettingsModalOpen(false)
    } catch (err: any) { setError(err.message) }
  }

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setUploadedFiles(files)
      const firstFile = files[0]
      const relativePath = (firstFile as any).webkitRelativePath || firstFile.name
      const folderName = relativePath.split('/')[0]
      setDatasetPath(folderName); setDatasetNameInput(folderName)
    }
  }

  const handleLoadDataset = async () => {
    if (uploadMode === 'local' && !uploadedFiles) return
    if (uploadMode === 's3' && (!s3AccessKey || !s3SecretKey || !s3Bucket || !s3UserPath)) return
    if (uploadMode === 'huggingface' && (!hfRepoId)) return

    // Auto-generate dataset name if not set
    let finalDatasetName = datasetNameInput || datasetName
    if (!finalDatasetName) {
      if (uploadMode === 'local' && uploadedFiles) {
        const firstFile = uploadedFiles[0]
        const relativePath = (firstFile as any).webkitRelativePath || firstFile.name
        finalDatasetName = relativePath.split('/')[0]
      } else if (uploadMode === 's3' && s3UserPath) {
        const pathParts = s3UserPath.split('/').filter(p => p)
        finalDatasetName = pathParts[pathParts.length - 1] || pathParts[pathParts.length - 2] || 'dataset'
      } else if (uploadMode === 'huggingface' && hfRepoId) {
        finalDatasetName = hfRepoId.split('/').pop() || 'dataset'
      } else {
        return
      }
    }

    setUploadLoading(true)
    setUploadSuccess(false)
    setError(null)

    try {
      let response: Response

      if (uploadMode === 'local') {
        const formData = new FormData()
        if (uploadedFiles) {
          Array.from(uploadedFiles).forEach((file) => {
            const relativePath = (file as any).webkitRelativePath || file.name
            formData.append('files', file, relativePath)
          })
        }
        formData.append('dataset_name', finalDatasetName)
        formData.append('environment', environment || '')
        formData.append('axes', JSON.stringify(selectedAxes.length > 0 ? selectedAxes : []))

        response = await fetch(`${API_BASE}/api/datasets/upload`, {
          method: 'POST',
          body: formData,
        })
      } else if (uploadMode === 's3') {
        response = await fetch(`${API_BASE}/api/datasets/upload-s3`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_name: finalDatasetName,
            s3_access_key: s3AccessKey,
            s3_secret_key: s3SecretKey,
            s3_bucket: s3Bucket,
            s3_region: s3Region,
            s3_path: s3UserPath,
            environment: environment || '',
            axes: selectedAxes.length > 0 ? selectedAxes : null,
          }),
        })
      } else if (uploadMode === 'huggingface') {
        response = await fetch(`${API_BASE}/api/datasets/upload-huggingface`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_name: finalDatasetName,
            hf_repo_id: hfRepoId,
            hf_split: hfSplit,
            hf_token: hfToken || null,
            environment: environment || '',
            axes: selectedAxes.length > 0 ? selectedAxes : null,
          }),
        })
      } else {
        throw new Error('Invalid upload mode')
      }

      if (!response.ok) {
        let errorMessage = 'Failed to load dataset'
        try {
          const contentType = response.headers.get('content-type')
          if (contentType && contentType.includes('application/json')) {
            const err = await response.json()
            errorMessage = err.detail || err.message || errorMessage
          } else {
            const text = await response.text()
            errorMessage = text || errorMessage
          }
        } catch {
          errorMessage = `Server error: ${response.status} ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      // Upload successful — navigate to the dataset page to trigger ingestion polling
      setUploadSuccess(true)
      setUploadLoading(false)
      router.push(`/dataset/${encodeURIComponent(finalDatasetName)}`)
    } catch (err: any) {
      setError(err.message)
      setUploadLoading(false)
    }
  }

  // ── Distribution loading ──
  useEffect(() => {
    if (!ingestionComplete || !datasetName) return
    if (selectedAxes.length === 0 && !environment) return
    if (selectedAxes.length === 0 && environment) return

    const loadDistributions = async () => {
      try {
        setDistributionsLoading(true)
        const params = new URLSearchParams()
        if (datasetName) params.append('dataset_name', datasetName)
        if (environment) params.append('environment', environment)
        params.append('axes', JSON.stringify(selectedAxes))
        if (analysisView === 'new') params.append('include_variations', 'true')
        else params.append('include_variations', 'false')

        const url = `${API_BASE}/api/ares/distributions?${params.toString()}`
        const distResponse = await fetch(url)
        if (distResponse.ok) {
          const distData = await distResponse.json()
          const vizs = distData.visualizations || []
          const vi = distData.variations_included || false
          if (analysisView === 'new') { if (vizs.length > 0) { setNewDistributions(vizs); setVariationsIncluded(vi) } else { setVariationsIncluded(false) } }
          else { if (vizs.length > 0) setAresDistributions(vizs) }
        }
      } catch (err: any) { console.error('Error loading distributions:', err) }
      finally { setDistributionsLoading(false) }
    }
    loadDistributions()
  }, [ingestionComplete, datasetName, environment, selectedAxes, analysisView])

  // ── Ingestion status polling ──
  // When navigating to a task, try to load data. If the backend returns 500
  // (e.g. database not found), skip straight to the empty/no-data state
  // instead of retrying in a loop.
  useEffect(() => {
    if (!datasetName) { setWaitingForIngestion(false); setIngestionComplete(true); return }
    let isMounted = true
    const checkIngestionStatus = async () => {
      if (!isMounted) return
      try {
        const params = new URLSearchParams()
        params.append('dataset_name', datasetName); params.append('environment', 'Indoor')
        params.append('axes', JSON.stringify(['Objects', 'Lighting', 'Color/Material']))
        const url = `${API_BASE}/api/ares/distributions?${params.toString()}`
        const distResponse = await fetch(url)
        if (distResponse.ok) {
          const distData = await distResponse.json()
          if (distData.ingestion_status === 'in_progress') {
            // Still ingesting — poll again
            setTimeout(checkIngestionStatus, 1000)
            return
          }
          // Ingestion complete or not applicable — load dataset
          if (isMounted) { setIngestionProgress(100); setIngestionComplete(true); setWaitingForIngestion(false); handleDatasetLoaded(datasetName) }
        } else {
          // Server error (500) or not-found — proceed to empty state, no retry loop
          console.warn(`Distributions endpoint returned ${distResponse.status} for ${datasetName}, proceeding to empty state`)
          if (isMounted) { setIngestionProgress(100); setIngestionComplete(true); setWaitingForIngestion(false); setLoading(false) }
        }
      } catch (err) {
        // Network error — proceed to empty state
        console.warn('Failed to check ingestion status:', err)
        if (isMounted) { setIngestionProgress(100); setIngestionComplete(true); setWaitingForIngestion(false); setLoading(false) }
      }
    }
    setWaitingForIngestion(true); setIngestionComplete(false); setIngestionProgress(0); checkIngestionStatus()
    return () => { isMounted = false }
  }, [datasetName])

  // ── Environment / axes initialization from URL params ──
  const [axesInitializedFromUrl, setAxesInitializedFromUrl] = useState(false)
  useEffect(() => {
    const envParam = searchParams?.get('environment'); const axesParam = searchParams?.get('axes')
    if (envParam) { if (envParam === 'Indoor') { setIsIndoor(true); setIsOutdoor(false) } else if (envParam === 'Outdoor') { setIsIndoor(false); setIsOutdoor(true) } }
    if (axesParam) { try { const p = JSON.parse(axesParam); if (Array.isArray(p)) { setSelectedAxes(p); setAxesInitializedFromUrl(true) } } catch { } }
  }, [searchParams])

  useEffect(() => {
    const envParam = searchParams?.get('environment')
    if (ingestionComplete && !envParam && !environment && !isIndoor && !isOutdoor) setIsIndoor(true)
  }, [ingestionComplete, environment, isIndoor, isOutdoor, searchParams])

  const [prevEnvironment, setPrevEnvironment] = useState<string>('')
  useEffect(() => {
    if (environment !== prevEnvironment && !axesInitializedFromUrl) {
      if (environment === 'Indoor') setSelectedAxes(['Objects', 'Lighting', 'Color/Material'])
      else if (environment === 'Outdoor') setSelectedAxes(['Objects', 'Lighting', 'Weather', 'Road Surface'])
      else setSelectedAxes([])
    }
    if (environment !== prevEnvironment) setPrevEnvironment(environment)
  }, [environment, prevEnvironment, axesInitializedFromUrl])

  useEffect(() => {
    if (isIndoor && !isOutdoor) setEnvironment('Indoor')
    else if (isOutdoor && !isIndoor) setEnvironment('Outdoor')
    else if (!isIndoor && !isOutdoor) setEnvironment('')
  }, [isIndoor, isOutdoor])

  // ── Auth guards ──
  if (authLoading) return <div className="min-h-screen bg-[#1a1a1a] flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-[#9aa4b5]" /></div>
  if (!isAuthenticated) return (<><LoginModal isOpen={isLoginModalOpen} onClose={() => setIsLoginModalOpen(false)} onSwitchToRegister={() => { setIsLoginModalOpen(false); setIsRegisterModalOpen(true) }} /><RegisterModal isOpen={isRegisterModalOpen} onClose={() => setIsRegisterModalOpen(false)} onSwitchToLogin={() => { setIsRegisterModalOpen(false); setIsLoginModalOpen(true) }} /></>)
  if (waitingForIngestion || !ingestionComplete) return (<div className="min-h-screen bg-[#1a1a1a] flex items-center justify-center"><div className="w-full max-w-md px-8"><div className="h-1.5 bg-[#2a2a2a] rounded-full overflow-hidden"><div className="h-full bg-[#9aa4b5] rounded-full transition-all duration-500 ease-out" style={{ width: `${ingestionProgress}%` }} /></div></div></div>)

  // ── Sidebar metrics ──
  const hasNoData = !datasetInfo && !loading

  return (
    <div className="min-h-screen bg-[#1a1a1a] text-white">
      <SideMenu isOpen={isSideMenuOpen} onToggle={() => setIsSideMenuOpen(!isSideMenuOpen)}
        onAddTask={() => router.push('/')} onOpenSettings={() => setIsSettingsModalOpen(true)}
        onOpenBilling={() => setIsBillingModalOpen(true)}
        onLogout={() => { logout(); setIsLoginModalOpen(true); router.push('/') }}
        tasks={SEED_TASK_NAMES.map(n => ({ name: n }))} />

      <TaskModal isOpen={isTaskModalOpen} onClose={() => setIsTaskModalOpen(false)} onSave={handleSaveTask} />
      <SettingsModal isOpen={isSettingsModalOpen} onClose={() => setIsSettingsModalOpen(false)} onSave={handleSaveSettings} />
      <BillingModal isOpen={isBillingModalOpen} onClose={() => setIsBillingModalOpen(false)} />

      {/* Header */}
      <header className="border-b border-[#2a2a2a] bg-[#222222] sticky top-0 z-50">
        <div className="flex items-center justify-between h-10">
          <div className="flex items-center h-full">
            <button onClick={() => setIsSideMenuOpen(!isSideMenuOpen)} className="px-4 h-full text-white/40 hover:text-white hover:bg-[#2a2a2a] transition-colors border-r border-[#2a2a2a]" aria-label="Toggle menu"><Menu className="w-4 h-4" /></button>
            <div className="flex items-center gap-4 px-6">
              <h1 className="text-sm font-medium tracking-wide text-white cursor-pointer hover:text-white/80" onClick={() => router.push('/')}>6d labs</h1>
              {datasetName && (<><div className="h-3 w-px bg-[#2a2a2a]" /><span className="text-xs text-white/40 font-mono">{decodeURIComponent(datasetName)}</span></>)}
            </div>
          </div>
          <div className="px-4">
            <button onClick={handleExportDataset} className="px-3 py-1.5 text-xs text-white bg-[#4b6671] hover:bg-[#3d5560] transition-colors rounded-lg">Export</button>
          </div>
        </div>
      </header>

      <div className="min-h-[calc(100vh-40px)]">
        {/* ── Main Content ── */}
        <main className="flex-1 flex flex-col min-h-0">
          {/* Tabs */}
          <div className="border-b border-[#2a2a2a] bg-[#1a1a1a] flex-shrink-0 z-40">
            <div className="flex gap-0 px-6">
              {TABS.map(tab => (
                <button key={tab.key} onClick={() => setActiveTab(tab.key)}
                  className={cn("px-4 py-3 text-xs font-medium transition-colors relative",
                    activeTab === tab.key ? 'text-white' : 'text-white/40 hover:text-white')}>
                  {tab.label}
                  {activeTab === tab.key && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#4b6671]" />}
                </button>
              ))}
            </div>
          </div>

          {/* Scrollable content area */}
          <div className="flex-1 overflow-y-auto">
            {/* Error */}
            {error && (<div className="mx-6 mt-4 p-2.5 bg-[#2a1a1a] border border-[#3a2a2a] rounded-xl flex items-center gap-2"><XCircle className="w-3.5 h-3.5 text-[#cc6666]" /><span className="text-xs text-[#cc6666]">{error}</span></div>)}

            {/* Loading */}
            {loading && <div className="flex items-center justify-center py-12"><Loader2 className="w-5 h-5 animate-spin text-white/40" /></div>}

            {/* Tab Content */}
            {!loading && (
              <div className="p-6">
                {activeTab === 'overview' && <OverviewTab datasetName={datasetName} datasetInfo={datasetInfo} aresDistributions={aresDistributions} newDistributions={newDistributions} distributionsLoading={distributionsLoading} analysisView={analysisView} setAnalysisView={setAnalysisView} analysisSwitchEnabled={analysisSwitchEnabled} variationsIncluded={variationsIncluded} hasNoData={hasNoData} uploadMode={uploadMode} setUploadMode={setUploadMode} uploadedFiles={uploadedFiles} datasetPath={datasetPath} handleFolderSelect={handleFolderSelect} s3AccessKey={s3AccessKey} setS3AccessKey={setS3AccessKey} s3SecretKey={s3SecretKey} setS3SecretKey={setS3SecretKey} s3Bucket={s3Bucket} setS3Bucket={setS3Bucket} s3Region={s3Region} setS3Region={setS3Region} s3UserPath={s3UserPath} setS3UserPath={setS3UserPath} hfRepoId={hfRepoId} setHfRepoId={setHfRepoId} hfSplit={hfSplit} setHfSplit={setHfSplit} hfToken={hfToken} setHfToken={setHfToken} handleLoadDataset={handleLoadDataset} uploadLoading={uploadLoading} uploadSuccess={uploadSuccess} setDatasetPath={setDatasetPath} setDatasetName={setDatasetNameInput} isIndoor={isIndoor} setIsIndoor={setIsIndoor} isOutdoor={isOutdoor} setIsOutdoor={setIsOutdoor} selectedAxes={selectedAxes} setSelectedAxes={setSelectedAxes} />}
                {activeTab === 'actions' && <ActionsTab datasetName={datasetName} datasetData={datasetData} onAugmentationComplete={handleAugmentationComplete} uploadMode={uploadMode} setUploadMode={setUploadMode} uploadedFiles={uploadedFiles} datasetPath={datasetPath} handleFolderSelect={handleFolderSelect} s3AccessKey={s3AccessKey} setS3AccessKey={setS3AccessKey} s3SecretKey={s3SecretKey} setS3SecretKey={setS3SecretKey} s3Bucket={s3Bucket} setS3Bucket={setS3Bucket} s3Region={s3Region} setS3Region={setS3Region} s3UserPath={s3UserPath} setS3UserPath={setS3UserPath} hfRepoId={hfRepoId} setHfRepoId={setHfRepoId} hfSplit={hfSplit} setHfSplit={setHfSplit} hfToken={hfToken} setHfToken={setHfToken} handleLoadDataset={handleLoadDataset} uploadLoading={uploadLoading} uploadSuccess={uploadSuccess} setDatasetPath={setDatasetPath} setDatasetName={setDatasetNameInput} isIndoor={isIndoor} setIsIndoor={setIsIndoor} isOutdoor={isOutdoor} setIsOutdoor={setIsOutdoor} selectedAxes={selectedAxes} setSelectedAxes={setSelectedAxes} />}
                {activeTab === 'history' && <HistoryTab />}
              </div>
            )}
          </div>
        </main>
      </div>

      <LoginModal isOpen={isLoginModalOpen} onClose={() => setIsLoginModalOpen(false)} onSwitchToRegister={() => { setIsLoginModalOpen(false); setIsRegisterModalOpen(true) }} />
      <RegisterModal isOpen={isRegisterModalOpen} onClose={() => setIsRegisterModalOpen(false)} onSwitchToLogin={() => { setIsRegisterModalOpen(false); setIsLoginModalOpen(true) }} />
    </div>
  )
}
