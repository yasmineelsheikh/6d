'use client'

import { useState, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { Loader2, XCircle, Menu, ChevronDown, ChevronRight } from 'lucide-react'
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

interface DatasetInfo {
  dataset_name: string
  total_episodes: number
  robot_type: string
}

interface AresState {
  total_rows: number
  total_statistics: any
  annotation_statistics: any
  has_embeddings: boolean
  has_annotations: boolean
}

interface Visualization {
  title: string
  figure: any
}

export default function DatasetPage() {
  const params = useParams()
  const router = useRouter()
  const datasetName = params?.name as string

  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [datasetData, setDatasetData] = useState<any[]>([])
  const [curatedData, setCuratedData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // ARES state
  const [aresInitialized, setAresInitialized] = useState(false)
  const [aresLoading, setAresLoading] = useState(false)
  const [aresError, setAresError] = useState<string | null>(null)
  const [aresState, setAresState] = useState<AresState | null>(null)
  
  // Distributions state
  const [aresDistributions, setAresDistributions] = useState<Visualization[]>([])
  
  // Authentication
  const { user, isAuthenticated, loading: authLoading, logout } = useAuth()
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false)
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false)
  
  // Plot configuration state
  const [environment, setEnvironment] = useState<'Indoor' | 'Outdoor' | ''>('')
  const [selectedAxes, setSelectedAxes] = useState<string[]>([])
  const [showAdvancedAxes, setShowAdvancedAxes] = useState(false)
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
      const response = await fetch(`/api/datasets/${name}/info`)
      if (!response.ok) throw new Error('Failed to load dataset info')
      const info = await response.json()
      setDatasetInfo(info)
      
      const dataResponse = await fetch(`/api/datasets/${name}/data`)
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
      const dataResponse = await fetch(`/api/datasets/${datasetName}/data`)
      if (!dataResponse.ok) throw new Error('Failed to load curated data')
      const data = await dataResponse.json()
      setCuratedData(data.data || [])
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleExportDataset = async () => {
    if (!datasetName) return
    try {
      const response = await fetch(`/api/datasets/${datasetName}/export`)
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
      const response = await fetch('/api/tasks', {
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
      const response = await fetch('/api/settings', {
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

  // Initialize ARES data
  useEffect(() => {
    const init = async () => {
      setAresLoading(true)
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 60000)
        
        const response = await fetch('/api/ares/initialize', {
          method: 'POST',
          signal: controller.signal,
        })
        clearTimeout(timeoutId)
        
        if (!response.ok) {
          let errorMessage = `Failed to initialize ARES: ${response.status} ${response.statusText}`
          try {
            const text = await response.text()
            try {
              const errorData = JSON.parse(text)
              errorMessage = errorData.detail || errorData.message || text || errorMessage
            } catch {
              errorMessage = text || errorMessage
            }
          } catch (e) {
            // Keep default error message
          }
          throw new Error(errorMessage)
        }
        
        const initData = await response.json()
        const stateResponse = await fetch('/api/ares/state')
        if (!stateResponse.ok) {
          throw new Error('Failed to get state')
        }
        const stateData = await stateResponse.json()
        setAresState(stateData)
        setAresInitialized(true)
      } catch (err: any) {
        let errorMsg = err.message || err.toString() || 'Unknown error occurred'
        if (err.name === 'AbortError') {
          errorMsg = 'Initialization timed out after 60 seconds.'
        }
        setAresError(errorMsg)
      } finally {
        setAresLoading(false)
      }
    }
    init()
  }, [])

  // Load distributions
  useEffect(() => {
    if (!aresInitialized) return

    const loadDistributions = async () => {
      try {
        const distResponse = await fetch('/api/ares/distributions')
        if (distResponse.ok) {
          const distData = await distResponse.json()
          const vizs = distData.visualizations || []
          setAresDistributions(vizs)
        }
      } catch (err: any) {
        console.error('Error loading distributions:', err)
      }
    }
    loadDistributions()
  }, [aresInitialized])

  // Load dataset when component mounts or dataset name changes
  useEffect(() => {
    if (datasetName) {
      handleDatasetLoaded(datasetName)
    }
  }, [datasetName])

  // Initialize all axes as checked by default when environment changes
  useEffect(() => {
    if (environment === 'Indoor') {
      setSelectedAxes(['Objects', 'Lighting', 'Color/Material'])
    } else if (environment === 'Outdoor') {
      setSelectedAxes(['Objects', 'Lighting', 'Weather', 'Road Surface'])
    } else {
      setSelectedAxes([])
    }
  }, [environment])
  
  // Update environment based on checkbox states
  useEffect(() => {
    if (isIndoor && !isOutdoor) {
      setEnvironment('Indoor')
    } else if (isOutdoor && !isIndoor) {
      setEnvironment('Outdoor')
    } else if (!isIndoor && !isOutdoor) {
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
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Data Analysis</h2>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6 space-y-8">
              <DatasetOverview datasetInfo={datasetInfo} />
              <DatasetDistributions 
                datasetName={datasetName} 
                aresDistributions={aresDistributions}
                aresInitialized={aresInitialized}
              />
              <EpisodePreview datasetData={datasetData} />
            </div>
          </div>
        )}

        {/* Dataset Curation Section */}
        {datasetName && !loading && (
          <div className="mb-8">
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Dataset Curation</h2>
            <div className="bg-[#222222] border border-[#2a2a2a]">
              <div className="p-6">
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                  <AugmentationPanel
                    datasetName={datasetName}
                    onComplete={handleAugmentationComplete}
                  />
                  <div className="mt-4 pt-4 border-t border-[#2a2a2a] opacity-50 pointer-events-none">
                    <div className="relative">
                      <OptimizationPanel
                        datasetName={datasetName}
                      />
                      <div className="absolute inset-0 flex items-center justify-center bg-[#1a1a1a] bg-opacity-80">
                        <span className="text-sm text-[#9aa4b5] font-medium">Coming Soon</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-[#2a2a2a] opacity-50 pointer-events-none">
                    <div className="relative">
                      <TestingPanel datasetName={datasetName} />
                      <div className="absolute inset-0 flex items-center justify-center bg-[#1a1a1a] bg-opacity-80">
                        <span className="text-sm text-[#9aa4b5] font-medium">Coming Soon</span>
                      </div>
                    </div>
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
