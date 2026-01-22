'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { Loader2, XCircle, Menu } from 'lucide-react'
import DatasetOverview from '@/components/DatasetOverview'
import DatasetDistributions from '@/components/DatasetDistributions'
import EpisodePreview from '@/components/EpisodePreview'
import AugmentationPanel from '@/components/AugmentationPanel'
import OptimizationPanel from '@/components/OptimizationPanel'
import TestingPanel from '@/components/TestingPanel'
import SideMenu from '@/components/SideMenu'
import TaskModal, { TaskData } from '@/components/TaskModal'
import SettingsModal, { SettingsData } from '@/components/SettingsModal'
import LoginModal from '@/components/LoginModal'
import RegisterModal from '@/components/RegisterModal'
import { useAuth } from '@/contexts/AuthContext'
import { cn } from '@/lib/utils'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

interface DatasetInfo {
  dataset_name: string
  total_episodes: number
  robot_type: string
}

interface Visualization {
  title: string
  figure: any
}

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
  
  // Distributions state
  const [aresDistributions, setAresDistributions] = useState<Visualization[]>([])
  
  // Authentication
  const { user, isAuthenticated, loading: authLoading, logout } = useAuth()
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false)
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false)
  
  // Plot configuration state
  const [environment, setEnvironment] = useState<'Indoor' | 'Outdoor' | ''>('')
  const [selectedAxes, setSelectedAxes] = useState<string[]>([])
  const [isIndoor, setIsIndoor] = useState(false)
  const [isOutdoor, setIsOutdoor] = useState(false)
  
  // Side menu and modals state
  const [isSideMenuOpen, setIsSideMenuOpen] = useState(false)
  const [isTaskModalOpen, setIsTaskModalOpen] = useState(false)
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false)
  const [tasks, setTasks] = useState<TaskData[]>([])

  const handleDatasetLoaded = async (name: string) => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_BASE}/api/datasets/${name}/info`)
      if (!response.ok) throw new Error('Failed to load dataset info')
      const info = await response.json()
      setDatasetInfo(info)
      
      const dataResponse = await fetch(`${API_BASE}/api/datasets/${name}/data`)
      if (!dataResponse.ok) throw new Error('Failed to load dataset data')
      const data = await dataResponse.json()
      setDatasetData(data.data || [])
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAugmentationComplete = async (datasetName: string) => {
    // Reload dataset data for curated view
    try {
      const dataResponse = await fetch(`${API_BASE}/api/datasets/${datasetName}/data`)
      if (!dataResponse.ok) throw new Error('Failed to load curated data')
      const data = await dataResponse.json()
      setCuratedData(data.data || [])
      setAnalysisSwitchEnabled(true)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleExportDataset = async () => {
    if (!datasetName) return
    try {
      const response = await fetch(`${API_BASE}/api/datasets/${datasetName}/export`)
      if (!response.ok) throw new Error('Failed to export dataset')
      const data = await response.json()
      
      // Create download link
      const blob = new Blob([data.content], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = data.filename || `${datasetName}.csv`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleSaveTask = async (task: TaskData) => {
    try {
      const response = await fetch(`${API_BASE}/api/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(task),
      })
      if (!response.ok) throw new Error('Failed to save task')
      const savedTask = await response.json()
      setTasks([...tasks, savedTask])
      setIsTaskModalOpen(false)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleSaveSettings = async (settings: SettingsData) => {
    try {
      const response = await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      })
      if (!response.ok) throw new Error('Failed to save settings')
      setIsSettingsModalOpen(false)
    } catch (err: any) {
      setError(err.message)
    }
  }

  // Load distributions - only after ingestion is complete
  useEffect(() => {
    // Don't load distributions if ingestion is not complete or dataset is not loaded
    if (!ingestionComplete || !datasetName) {
      // Only clear distributions if we don't have valid data yet
      // Don't clear if we already have distributions loaded
      return
    }

    // Don't make API call if no axes are selected AND no environment is set
    // If environment is set but selectedAxes is empty, it means selectedAxes is being updated
    // in another useEffect, so we'll wait for that update to trigger this effect again
    // Only skip if both are empty (user hasn't selected anything yet)
    if (selectedAxes.length === 0 && !environment) {
      console.log('[FRONTEND] Skipping distributions call - no axes and no environment selected. Existing vizs:', aresDistributions.length)
      // Don't clear existing distributions - just skip the call
      return
    }
    
    // If environment is set but axes are empty, this is likely a race condition where
    // environment changed but selectedAxes hasn't been updated yet. Skip this call
    // and wait for selectedAxes to update (which will trigger this effect again)
    if (selectedAxes.length === 0 && environment) {
      console.log('[FRONTEND] Waiting for selectedAxes to update - Environment:', environment, 'Existing vizs:', aresDistributions.length)
      return
    }
    
    console.log('[FRONTEND] Making distributions call - Environment:', environment, 'Axes:', selectedAxes, 'Existing vizs:', aresDistributions.length)

    const loadDistributions = async () => {
      try {
        setDistributionsLoading(true)
        
        // Build query parameters
        const params = new URLSearchParams()
        if (datasetName) {
          params.append('dataset_name', datasetName)
        }
        if (environment) {
          params.append('environment', environment)
        }
        // Always send axes parameter, even if empty, so backend knows user's selection
        params.append('axes', JSON.stringify(selectedAxes))
        
        const url = `${API_BASE}/api/ares/distributions${params.toString() ? '?' + params.toString() : ''}`
        const distResponse = await fetch(url)
        if (distResponse.ok) {
          const distData = await distResponse.json()
          const vizs = distData.visualizations || []
          console.log('[FRONTEND] Received distributions:', vizs.length, 'visualizations')
          console.log('[FRONTEND] Distribution titles:', vizs.map((v: any) => v.title))
          console.log('[FRONTEND] Current state before update:', aresDistributions.length, 'existing visualizations')
          // Only update state if we got visualizations, or if we explicitly want to clear
          // This prevents empty responses from overwriting good visualizations
          if (vizs.length > 0) {
            setAresDistributions(vizs)
          } else {
            console.log('[FRONTEND] Skipping state update - received 0 visualizations, keeping existing:', aresDistributions.length)
            // Don't overwrite existing distributions with empty array
          }
        } else {
          console.error('Failed to load distributions:', distResponse.status, distResponse.statusText)
        }
      } catch (err: any) {
        console.error('Error loading distributions:', err)
      } finally {
        setDistributionsLoading(false)
      }
    }
    
    loadDistributions()
  }, [ingestionComplete, datasetName, environment, selectedAxes])

  // Check ingestion status and wait for completion before showing page
  useEffect(() => {
    if (!datasetName) {
      setWaitingForIngestion(false)
      setIngestionComplete(true)
      return
    }

    let pollInterval: NodeJS.Timeout | null = null
    let progressInterval: NodeJS.Timeout | null = null
    let retryCount = 0
    const maxRetries = 300 // Poll for up to 5 minutes (1 second intervals)
    const pollIntervalMs = 1000 // Poll every 1 second
    let isMounted = true
    const startTime = Date.now()
    const estimatedMaxTime = 300000 // 5 minutes in milliseconds

    // Update progress gradually while waiting
    const updateProgress = () => {
      if (!isMounted) return
      
      const elapsed = Date.now() - startTime
      // Gradually increase progress, but cap at 90% until ingestion completes
      const timeBasedProgress = Math.min(90, (elapsed / estimatedMaxTime) * 90)
      setIngestionProgress(timeBasedProgress)
    }

    const checkIngestionStatus = async () => {
      if (!isMounted) return

      try {
        // Check ingestion status via distributions endpoint
        const params = new URLSearchParams()
        params.append('dataset_name', datasetName)
        // Use default Indoor environment to check status
        params.append('environment', 'Indoor')
        params.append('axes', JSON.stringify(['Objects', 'Lighting', 'Color/Material']))
        
        const url = `${API_BASE}/api/ares/distributions?${params.toString()}`
        const distResponse = await fetch(url)
        
        if (distResponse.ok) {
          const distData = await distResponse.json()
          
          // If ingestion is in progress, keep polling
          if (distData.ingestion_status === 'in_progress') {
            console.log('Ingestion in progress, waiting for completion...')
            if (!pollInterval && retryCount < maxRetries) {
              // Start progress updates
              if (!progressInterval) {
                progressInterval = setInterval(updateProgress, 100) // Update every 100ms
              }
              
              // Start polling
              pollInterval = setInterval(() => {
                if (!isMounted) {
                  if (pollInterval) clearInterval(pollInterval)
                  if (progressInterval) clearInterval(progressInterval)
                  return
                }
                retryCount++
                if (retryCount >= maxRetries) {
                  if (pollInterval) clearInterval(pollInterval)
                  if (progressInterval) clearInterval(progressInterval)
                  pollInterval = null
                  progressInterval = null
                  if (isMounted) {
                    setIngestionProgress(100)
                    setWaitingForIngestion(false)
                    setIngestionComplete(true) // Show page even if timeout
                  }
                  console.log('Max retries reached, showing page anyway')
                  return
                }
                checkIngestionStatus()
              }, pollIntervalMs)
            }
            return
          }
          
          // Ingestion is complete (or not in progress)
          if (pollInterval) {
            clearInterval(pollInterval)
            pollInterval = null
          }
          if (progressInterval) {
            clearInterval(progressInterval)
            progressInterval = null
          }
          
          if (isMounted) {
            setIngestionProgress(100)
            setIngestionComplete(true)
            setWaitingForIngestion(false)
            console.log('Ingestion complete, showing page')
            // Load the dataset info (distributions will load automatically via useEffect)
            handleDatasetLoaded(datasetName)
          }
        } else {
          // If request failed, assume ingestion might be complete and show page
          if (retryCount < 10) {
            retryCount++
            setTimeout(() => {
              if (isMounted) checkIngestionStatus()
            }, pollIntervalMs)
          } else {
            // After 10 retries, assume ready and show page
            if (isMounted) {
              if (progressInterval) clearInterval(progressInterval)
              setIngestionProgress(100)
              setIngestionComplete(true)
              setWaitingForIngestion(false)
              handleDatasetLoaded(datasetName)
            }
          }
        }
      } catch (err: any) {
        console.error('Error checking ingestion status:', err)
        // On error, retry a few times then show page
        if (retryCount < 10) {
          retryCount++
          setTimeout(() => {
            if (isMounted) checkIngestionStatus()
          }, pollIntervalMs)
        } else {
          // After 10 retries, assume ready and show page
          if (isMounted) {
            if (progressInterval) clearInterval(progressInterval)
            setIngestionProgress(100)
            setIngestionComplete(true)
            setWaitingForIngestion(false)
            handleDatasetLoaded(datasetName)
          }
        }
      }
    }

    // Start checking ingestion status
    setWaitingForIngestion(true)
    setIngestionComplete(false)
    setIngestionProgress(0)
    checkIngestionStatus()

    // Cleanup on unmount
    return () => {
      isMounted = false
      if (pollInterval) {
        clearInterval(pollInterval)
      }
      if (progressInterval) {
        clearInterval(progressInterval)
      }
    }
  }, [datasetName])

  // Track if axes were initialized from URL params to prevent auto-selection from overriding them
  const [axesInitializedFromUrl, setAxesInitializedFromUrl] = useState(false)
  
  // Initialize environment and axes from URL query parameters (set during upload)
  useEffect(() => {
    const envParam = searchParams?.get('environment')
    const axesParam = searchParams?.get('axes')
    
    if (envParam) {
      console.log('[FRONTEND] Initializing from URL params - environment:', envParam, 'axes:', axesParam)
      if (envParam === 'Indoor') {
        setIsIndoor(true)
        setIsOutdoor(false)
      } else if (envParam === 'Outdoor') {
        setIsIndoor(false)
        setIsOutdoor(true)
      }
    }
    
    if (axesParam) {
      try {
        const parsedAxes = JSON.parse(axesParam)
        if (Array.isArray(parsedAxes)) {
          // Set axes from URL params - empty array is valid (user unchecked all)
          setSelectedAxes(parsedAxes)
          setAxesInitializedFromUrl(true)
          console.log('[FRONTEND] Initialized selectedAxes from URL:', parsedAxes)
        }
      } catch (e) {
        console.error('[FRONTEND] Failed to parse axes from URL:', e)
      }
    } else if (envParam) {
      // Environment was set but no axes param - means user didn't explicitly set axes
      // Don't mark as initialized from URL so auto-selection can run
      setAxesInitializedFromUrl(false)
    }
  }, [searchParams])

  // Default to Indoor environment when ingestion is complete and no environment is set
  // This matches the backend default and upload page settings
  // Only default if no environment was passed via URL params
  useEffect(() => {
    const envParam = searchParams?.get('environment')
    if (ingestionComplete && !envParam && !environment && !isIndoor && !isOutdoor) {
      console.log('[FRONTEND] Defaulting to Indoor environment (no URL params)')
      setIsIndoor(true)
    }
  }, [ingestionComplete, environment, isIndoor, isOutdoor, searchParams])

  // Initialize all axes as checked by default when environment changes
  // Only reset if environment actually changed AND axes weren't initialized from URL params
  // This prevents overriding user selections from the upload page
  const [prevEnvironment, setPrevEnvironment] = useState<string>('')
  useEffect(() => {
    // Only auto-select axes if environment actually changed AND axes weren't set from URL params
    if (environment !== prevEnvironment && !axesInitializedFromUrl) {
      if (environment === 'Indoor') {
        setSelectedAxes(['Objects', 'Lighting', 'Color/Material'])
      } else if (environment === 'Outdoor') {
        setSelectedAxes(['Objects', 'Lighting', 'Weather', 'Road Surface'])
      } else {
        setSelectedAxes([])
      }
      setPrevEnvironment(environment)
    } else if (environment !== prevEnvironment) {
      // Environment changed but axes were set from URL - just update prevEnvironment
      setPrevEnvironment(environment)
    }
  }, [environment, prevEnvironment, axesInitializedFromUrl])
  
  // Update environment based on checkbox states
  useEffect(() => {
    console.log('[FRONTEND] Environment useEffect - isIndoor:', isIndoor, 'isOutdoor:', isOutdoor)
    if (isIndoor && !isOutdoor) {
      console.log('[FRONTEND] Setting environment to Indoor')
      setEnvironment('Indoor')
    } else if (isOutdoor && !isIndoor) {
      console.log('[FRONTEND] Setting environment to Outdoor')
      setEnvironment('Outdoor')
    } else if (!isIndoor && !isOutdoor) {
      console.log('[FRONTEND] Clearing environment (no checkboxes selected)')
      setEnvironment('')
    }
  }, [isIndoor, isOutdoor])

  // Show loading/authentication modals
  if (authLoading) {
    return (
      <div className="min-h-screen bg-[#1a1a1a] flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-[#9aa4b5]" />
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <>
        <LoginModal
          isOpen={isLoginModalOpen}
          onClose={() => setIsLoginModalOpen(false)}
          onSwitchToRegister={() => {
            setIsLoginModalOpen(false)
            setIsRegisterModalOpen(true)
          }}
        />
        <RegisterModal
          isOpen={isRegisterModalOpen}
          onClose={() => setIsRegisterModalOpen(false)}
          onSwitchToLogin={() => {
            setIsRegisterModalOpen(false)
            setIsLoginModalOpen(true)
          }}
        />
      </>
    )
  }

  // Show loading screen while waiting for ingestion to complete
  if (waitingForIngestion || !ingestionComplete) {
    return (
      <div className="min-h-screen bg-[#1a1a1a] flex items-center justify-center">
        <div className="w-full max-w-md px-8">
          <div className="h-1.5 bg-[#2a2a2a] rounded-full overflow-hidden">
            <div 
              className="h-full bg-[#9aa4b5] rounded-full transition-all duration-500 ease-out"
              style={{
                width: `${ingestionProgress}%`
              }}
            />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#1a1a1a] text-[#d4d4d4]">
      {/* Side Menu */}
      <SideMenu
        isOpen={isSideMenuOpen}
        onToggle={() => setIsSideMenuOpen(!isSideMenuOpen)}
        onAddTask={() => router.push('/')}
        onOpenSettings={() => setIsSettingsModalOpen(true)}
        onLogout={() => {
          logout()
          setIsLoginModalOpen(true)
          setIsRegisterModalOpen(false)
          router.push('/')
        }}
      />

      {/* Task Modal */}
      <TaskModal
        isOpen={isTaskModalOpen}
        onClose={() => setIsTaskModalOpen(false)}
        onSave={handleSaveTask}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        onSave={handleSaveSettings}
      />

      {/* Header */}
      <header className="border-b border-[#2a2a2a] bg-[#222222] sticky top-0 z-50">
        <div className="flex items-center justify-between h-10">
          <div className="flex items-center h-full">
            <button
              onClick={() => setIsSideMenuOpen(!isSideMenuOpen)}
              className="px-4 h-full text-[#d4d4d4] hover:text-white hover:bg-[#2a2a2a] transition-colors border-r border-[#2a2a2a]"
              aria-label="Toggle menu"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-4 px-6 flex-1">
              <h1 className="text-sm font-medium tracking-wide text-white">
                6d labs
              </h1>
              {datasetName && (
                <>
                  <div className="h-3 w-px bg-[#2a2a2a]" />
                  <span className="text-xs text-[#b5becb] font-mono">
                    {datasetName}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-4 p-2.5 bg-[#2a1a1a] border border-[#3a2a2a] flex items-center gap-2">
            <XCircle className="w-3.5 h-3.5 text-[#cc6666]" />
            <span className="text-xs text-[#cc6666]">{error}</span>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-5 h-5 animate-spin text-[#9aa4b5]" />
          </div>
        )}

        {/* Data Analysis Section */}
        {datasetName && !loading && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-medium text-[#d4d4d4]">Data Analysis</h2>
              <div className="flex items-center gap-3">
                <div
                  className={`inline-flex rounded border border-[#2a2a2a] text-[11px] ${
                    !analysisSwitchEnabled ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                <button
                  type="button"
                  onClick={() => analysisSwitchEnabled && setAnalysisView('original')}
                  className={`px-2.5 py-1 transition-colors ${
                    analysisView === 'original'
                      ? 'bg-[#3b4454] text-[#e3e8f0]'
                      : 'bg-transparent text-[#9aa4b5]'
                  }`}
                >
                  Original
                </button>
                <button
                  type="button"
                  onClick={() => analysisSwitchEnabled && setAnalysisView('new')}
                  className={`px-2.5 py-1 transition-colors border-l border-[#2a2a2a] ${
                    analysisView === 'new'
                      ? 'bg-[#3b4454] text-[#e3e8f0]'
                      : 'bg-transparent text-[#9aa4b5]'
                  }`}
                >
                  New
                </button>
                </div>
              </div>
            </div>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6 space-y-8">
              {analysisView === 'original' && (
                <>
                  <DatasetOverview datasetInfo={datasetInfo} />
                  <DatasetDistributions 
                    datasetName={datasetName} 
                    aresDistributions={aresDistributions}
                    loading={distributionsLoading}
                  />
                  <EpisodePreview datasetData={datasetData} />
                </>
              )}
              {analysisView === 'new' && (
                <>
                  {/* For now, show the same content as Original */}
                  <DatasetOverview datasetInfo={datasetInfo} />
                  <DatasetDistributions 
                    datasetName={datasetName} 
                    aresDistributions={aresDistributions}
                    loading={distributionsLoading}
                  />
                  <EpisodePreview datasetData={datasetData} />
                </>
              )}
            </div>
          </div>
        )}

        {/* Dataset Curation Section */}
        {datasetName && !loading && (
          <div className="mb-8">
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Dataset Curation</h2>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6 space-y-6">
              {/* Augmentation Card */}
              <div>
                <h3 className="text-xs font-medium mb-2 text-[#d4d4d4]">Augmentation</h3>
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                  <AugmentationPanel
                    datasetName={datasetName}
                    onComplete={handleAugmentationComplete}
                  />
                </div>
              </div>

              {/* Optimization Card - Coming Soon */}
              <div>
                <h3 className="text-xs font-medium mb-2 text-[#d4d4d4]">Optimization</h3>
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4 opacity-50 pointer-events-none relative">
                  <OptimizationPanel
                    datasetName={datasetName}
                  />
                  <div className="absolute inset-0 flex items-center justify-center bg-[#1a1a1a] bg-opacity-80">
                    <span className="text-sm text-[#9aa4b5] font-medium">Coming Soon</span>
                  </div>
                </div>
              </div>

              {/* Update with Test Data Card - Coming Soon */}
              <div>
                <h3 className="text-xs font-medium mb-2 text-[#d4d4d4]">Update with test data</h3>
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4 opacity-50 pointer-events-none relative">
                  <TestingPanel datasetName={datasetName} />
                  <div className="absolute inset-0 flex items-center justify-center bg-[#1a1a1a] bg-opacity-80">
                    <span className="text-sm text-[#9aa4b5] font-medium">Coming Soon</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Export Section */}
        {datasetName && !loading && (
          <div className="mt-4 flex justify-end">
            <button
              onClick={handleExportDataset}
              className="px-4 py-2 text-xs bg-[#4b6671] text-white hover:bg-[#3d5560] transition-colors border border-[#2a2a2a]"
            >
              Export Dataset (CSV)
            </button>
          </div>
        )}
      </main>

      {/* Login Modal */}
      <LoginModal
        isOpen={isLoginModalOpen}
        onClose={() => setIsLoginModalOpen(false)}
        onSwitchToRegister={() => {
          setIsLoginModalOpen(false)
          setIsRegisterModalOpen(true)
        }}
      />

      {/* Register Modal */}
      <RegisterModal
        isOpen={isRegisterModalOpen}
        onClose={() => setIsRegisterModalOpen(false)}
        onSwitchToLogin={() => {
          setIsRegisterModalOpen(false)
          setIsLoginModalOpen(true)
        }}
      />
    </div>
  )
}
