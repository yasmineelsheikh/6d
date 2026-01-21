'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Upload, Loader2, CheckCircle2, XCircle, Menu, ChevronDown, ChevronRight, Folder, Cloud } from 'lucide-react'
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
import dynamic from 'next/dynamic'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface DatasetInfo {
  dataset_name: string
  total_episodes: number
  robot_type: string
}

interface Visualization {
  title: string
  figure: any
}

export default function Home() {
  const [currentDataset, setCurrentDataset] = useState<string | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [datasetData, setDatasetData] = useState<any[]>([])
  const [curatedData, setCuratedData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Distributions state
  const [aresDistributions, setAresDistributions] = useState<Visualization[]>([])
  
  // Authentication
  const { user, isAuthenticated, loading: authLoading, logout } = useAuth()
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false)
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false)
  
  // Upload form state
  const router = useRouter()
  const [datasetPath, setDatasetPath] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [uploadLoading, setUploadLoading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [uploadMode, setUploadMode] = useState<'local' | 's3'>('local')
  const [uploadedFiles, setUploadedFiles] = useState<FileList | null>(null)
  const [s3Path, setS3Path] = useState('')
  
  // Plot configuration state
  const [environment, setEnvironment] = useState<'Indoor' | 'Outdoor' | ''>('')
  const [selectedAxes, setSelectedAxes] = useState<string[]>([])
  const [showAdvancedAxes, setShowAdvancedAxes] = useState(false)
  const [isIndoor, setIsIndoor] = useState(false)
  const [isOutdoor, setIsOutdoor] = useState(false)
  
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
  
  // Side menu and modals state
  const [isSideMenuOpen, setIsSideMenuOpen] = useState(false)
  const [isTaskModalOpen, setIsTaskModalOpen] = useState(false)
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false)
  const [tasks, setTasks] = useState<TaskData[]>([])
  
  const handleLoadDataset = async () => {
    if (uploadMode === 'local' && !uploadedFiles) return
    if (uploadMode === 's3' && !s3Path) return
    
    // Auto-generate dataset name if not set
    let finalDatasetName = datasetName
    if (!finalDatasetName) {
      if (uploadMode === 'local' && uploadedFiles) {
        const firstFile = uploadedFiles[0]
        const relativePath = (firstFile as any).webkitRelativePath || firstFile.name
        finalDatasetName = relativePath.split('/')[0]
      } else if (uploadMode === 's3' && s3Path) {
        const pathParts = s3Path.replace('s3://', '').split('/').filter(p => p)
        finalDatasetName = pathParts[pathParts.length - 1] || pathParts[pathParts.length - 2] || 'dataset'
      } else {
        return // Cannot proceed without a dataset name
      }
      setDatasetName(finalDatasetName)
    }

    setUploadLoading(true)
    setUploadSuccess(false)
    setError(null)

    try {
      let response: Response

      if (uploadMode === 'local') {
        // Upload files as FormData
        const formData = new FormData()
        if (uploadedFiles) {
          Array.from(uploadedFiles).forEach((file) => {
            // Preserve directory structure by using relative path
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
      } else {
        // S3 path
        response = await fetch(`${API_BASE}/api/datasets/load`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_path: s3Path,
            dataset_name: finalDatasetName,
            environment: environment || null,
            axes: selectedAxes.length > 0 ? selectedAxes : null,
            is_s3: true,
          }),
        })
      }

      if (!response.ok) {
        let errorMessage = 'Failed to load dataset'
        try {
          const contentType = response.headers.get('content-type')
          if (contentType && contentType.includes('application/json')) {
            const error = await response.json()
            errorMessage = error.detail || error.message || errorMessage
          } else {
            const text = await response.text()
            errorMessage = text || errorMessage
          }
        } catch (e) {
          errorMessage = `Server error: ${response.status} ${response.statusText}`
        }

        // Log full error context so we can see what's going wrong in production
        console.error('Upload error', {
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers.entries()),
          message: errorMessage,
        })

        throw new Error(errorMessage)
      }

      setUploadSuccess(true)
      // Navigate to dataset page with environment and axes as query parameters
      const params = new URLSearchParams()
      if (environment) {
        params.append('environment', environment)
      }
      if (selectedAxes.length > 0) {
        params.append('axes', JSON.stringify(selectedAxes))
      }
      const queryString = params.toString()
      const url = `/dataset/${encodeURIComponent(finalDatasetName)}${queryString ? '?' + queryString : ''}`
      router.push(url)
    } catch (error: any) {
      setError(error.message)
      setUploadLoading(false)
    }
  }

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setUploadedFiles(files)
      // Set dataset path to show selected folder name
      const firstFile = files[0]
      const relativePath = (firstFile as any).webkitRelativePath || firstFile.name
      const folderName = relativePath.split('/')[0]
      setDatasetPath(folderName)
      // Auto-generate dataset name from folder name
      setDatasetName(folderName)
    }
  }

  const handleDatasetLoaded = async (datasetName: string) => {
    setCurrentDataset(datasetName)
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`/api/datasets/${datasetName}/info`)
      if (!response.ok) throw new Error('Failed to load dataset info')
      const info = await response.json()
      setDatasetInfo(info)
      
      const dataResponse = await fetch(`/api/datasets/${datasetName}/data`)
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


  // Load distributions - only when a dataset is loaded
  useEffect(() => {
    // Don't load distributions if no dataset is loaded (e.g., after "New Task")
    if (!currentDataset) {
      setAresDistributions([])
      return
    }

    let pollInterval: NodeJS.Timeout | null = null
    let retryCount = 0
    const maxRetries = 60 // Poll for up to 60 seconds (1 second intervals)
    const pollIntervalMs = 1000 // Poll every 1 second
    let isMounted = true

    const loadDistributions = async () => {
      if (!isMounted) return
      
      try {
        // Build query parameters
        const params = new URLSearchParams()
        if (currentDataset) {
          params.append('dataset_name', currentDataset)
        }
        if (environment) {
          params.append('environment', environment)
        }
        // Always send axes parameter, even if empty, so backend knows user's selection
        params.append('axes', JSON.stringify(selectedAxes))
        
        const url = `/api/ares/distributions${params.toString() ? '?' + params.toString() : ''}`
        const distResponse = await fetch(url)
        if (distResponse.ok) {
          const distData = await distResponse.json()
          
          // If ingestion is in progress, poll until it completes
          if (distData.ingestion_status === 'in_progress') {
            console.log('Ingestion in progress, polling for completion...')
            if (!pollInterval && retryCount < maxRetries) {
              // Start polling
              pollInterval = setInterval(() => {
                if (!isMounted) {
                  if (pollInterval) clearInterval(pollInterval)
                  return
                }
                retryCount++
                if (retryCount >= maxRetries) {
                  if (pollInterval) clearInterval(pollInterval)
                  pollInterval = null
                  console.log('Max retries reached, stopping poll')
                  return
                }
                loadDistributions()
              }, pollIntervalMs)
            }
            return
          }
          
          // Clear polling if ingestion completed
          if (pollInterval) {
            clearInterval(pollInterval)
            pollInterval = null
          }
          
          // Set distributions
          const vizs = distData.visualizations || []
          if (isMounted) {
            setAresDistributions(vizs)
            if (retryCount > 0) {
              console.log('Distributions loaded after ingestion completed')
            }
          }
        } else {
          // If request failed and we haven't retried too many times, retry
          if (retryCount < maxRetries) {
            retryCount++
            setTimeout(() => {
              if (isMounted) loadDistributions()
            }, pollIntervalMs)
          } else {
            console.error('Failed to load distributions:', distResponse.status, distResponse.statusText)
          }
        }
      } catch (err: any) {
        console.error('Error loading distributions:', err)
        // Retry on error if we haven't exceeded max retries
        if (retryCount < maxRetries) {
          retryCount++
          setTimeout(() => {
            if (isMounted) loadDistributions()
          }, pollIntervalMs)
        }
      }
    }

    loadDistributions()
    
    // Cleanup polling on unmount or dependency change
    return () => {
      isMounted = false
      if (pollInterval) {
        clearInterval(pollInterval)
      }
    }
  }, [currentDataset, environment, selectedAxes])

  // Load tasks from localStorage
  useEffect(() => {
    const savedTasks = localStorage.getItem('app_tasks')
    if (savedTasks) {
      try {
        setTasks(JSON.parse(savedTasks))
      } catch (e) {
        console.error('Failed to parse saved tasks:', e)
      }
    }
  }, [])

  // Handle task save
  const handleSaveTask = async (taskData: TaskData) => {
    const newTask = {
      ...taskData,
      id: Date.now().toString(),
      created_at: new Date().toISOString(),
    }
    const updatedTasks = [...tasks, newTask]
    setTasks(updatedTasks)
    localStorage.setItem('app_tasks', JSON.stringify(updatedTasks))
    
    // Optionally send to backend
    try {
      await fetch(`${API_BASE}/api/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTask),
      })
    } catch (err) {
      console.error('Failed to save task to backend:', err)
      // Continue anyway - task is saved locally
    }
  }

  // Handle settings save
  const handleSaveSettings = async (settingsData: SettingsData) => {
    // Settings are already saved to localStorage in the modal
    // Optionally send to backend
    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settingsData),
      })
    } catch (err) {
      console.error('Failed to save settings to backend:', err)
      // Continue anyway - settings are saved locally
    }
  }

  // Handle dataset export
  const handleExportDataset = async () => {
    if (!currentDataset) return
    try {
      const response = await fetch(`/api/datasets/${currentDataset}/export?format=csv`)
      if (!response.ok) {
        throw new Error('Failed to export dataset')
      }
      const data = await response.json()
      const blob = new Blob([data.content], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = data.filename || `${currentDataset}.csv`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export error:', err)
      alert('Failed to export dataset')
    }
  }

  // Show login modal if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      setIsLoginModalOpen(true)
    }
  }, [authLoading, isAuthenticated])

  // Show loading screen while checking auth
  if (authLoading) {
    return (
      <div className="min-h-screen bg-[#1a1a1a] flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-[#4b6671]" />
      </div>
    )
  }

  // Show login/register if not authenticated
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
        onAddTask={async () => {
          try {
            // Clear the database and initialize components when starting a new task
            const response = await fetch(`${API_BASE}/api/database/clear`, {
              method: 'POST',
            })
            if (!response.ok) {
              console.error('Failed to clear database')
            } else {
              const data = await response.json()
              console.log('Database cleared:', data.message)
            }
          } catch (error) {
            console.error('Error clearing database:', error)
          }
          // Reload the page
          window.location.reload()
        }}
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
              {currentDataset && (
                <>
                  <div className="h-3 w-px bg-[#2a2a2a]" />
                  <span className="text-xs text-[#b5becb] font-mono">
                    {currentDataset}
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Data Section */}
        <div className="mb-8">
          <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Upload Data</h2>
          <div className="bg-[#222222] border border-[#2a2a2a] p-6">
            <div className="flex items-start gap-4 mb-4">
              {/* Input boxes */}
              <div className="flex-1 flex items-center gap-4">
                {/* Upload Mode Toggle */}
                <div className="flex items-center gap-2 border border-[#2a2a2a] rounded p-1 bg-[#1a1a1a]">
                  <button
                    type="button"
                    onClick={() => {
                      setUploadMode('local')
                      setUploadedFiles(null)
                      setDatasetPath('')
                      setS3Path('')
                    }}
                    className={`px-3 py-1.5 text-xs flex items-center gap-1.5 transition-colors ${
                      uploadMode === 'local'
                        ? 'bg-[#4b6671] text-white'
                        : 'text-[#9aa4b5] hover:text-[#d4d4d4]'
                    }`}
                  >
                    <Folder className="w-3 h-3" />
                    Local
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setUploadMode('s3')
                      setUploadedFiles(null)
                      setDatasetPath('')
                    }}
                    className={`px-3 py-1.5 text-xs flex items-center gap-1.5 transition-colors ${
                      uploadMode === 's3'
                        ? 'bg-[#4b6671] text-white'
                        : 'text-[#9aa4b5] hover:text-[#d4d4d4]'
                    }`}
                  >
                    <Cloud className="w-3 h-3" />
                    S3
                  </button>
                </div>

                {/* Local Upload */}
                {uploadMode === 'local' && (
                  <div className="relative w-[576px]">
                    <input
                      type="file"
                      id="folder-upload"
                      {...({ webkitdirectory: '', directory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
                      multiple
                      onChange={handleFolderSelect}
                      className="hidden"
                    />
                    <label
                      htmlFor="folder-upload"
                      className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666666] text-xs focus:outline-none focus:border-[#3a3a3a] transition-colors cursor-pointer flex items-center gap-2 hover:bg-[#252525]"
                    >
                      <Folder className="w-4 h-4 flex-shrink-0" />
                      <span className="flex-1 truncate">
                        {datasetPath || 'Select folder...'}
                      </span>
                      {uploadedFiles && (
                        <span className="text-[#9aa4b5] text-[10px]">
                          ({uploadedFiles.length} files)
                        </span>
                      )}
                    </label>
                  </div>
                )}

                {/* S3 Path Input */}
                {uploadMode === 's3' && (
                  <input
                    type="text"
                    placeholder="s3://bucket-name/path/to/dataset"
                    value={s3Path}
                    onChange={(e) => {
                      const path = e.target.value
                      setS3Path(path)
                      setDatasetPath(path)
                      // Auto-generate dataset name from S3 path (last part)
                      if (path) {
                        const pathParts = path.replace('s3://', '').split('/').filter(p => p)
                        const datasetNameFromPath = pathParts[pathParts.length - 1] || pathParts[pathParts.length - 2] || 'dataset'
                        setDatasetName(datasetNameFromPath)
                      } else {
                        setDatasetName('')
                      }
                    }}
                    className="w-[576px] px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666666] text-xs focus:outline-none focus:border-[#3a3a3a] transition-colors"
                  />
                )}
              </div>
              
              {/* Settings on far right in fixed-width column */}
              <div className="flex items-center gap-3 flex-shrink-0 w-72 justify-end relative">
                {/* Environment Selection - Checkboxes */}
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={isIndoor}
                      onChange={(e) => {
                        setIsIndoor(e.target.checked)
                        if (e.target.checked) {
                          setIsOutdoor(false)
                        }
                      }}
                      className="w-3 h-3 accent-[#4b6671] cursor-pointer"
                    />
                    <span className="text-xs text-[#d4d4d4]">Indoor</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={isOutdoor}
                      onChange={(e) => {
                        setIsOutdoor(e.target.checked)
                        if (e.target.checked) {
                          setIsIndoor(false)
                        }
                      }}
                      className="w-3 h-3 accent-[#4b6671] cursor-pointer"
                    />
                    <span className="text-xs text-[#d4d4d4]">Outdoor</span>
                  </label>
                </div>

                {/* Advanced Axes Toggle - inline with Indoor/Outdoor */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowAdvancedAxes(!showAdvancedAxes)}
                    className="flex items-center gap-1 text-xs text-[#8a8a8a] hover:text-[#d4d4d4] transition-colors"
                  >
                    {showAdvancedAxes ? (
                      <ChevronDown className="w-3 h-3" />
                    ) : (
                      <ChevronRight className="w-3 h-3" />
                    )}
                    <span>Axes</span>
                  </button>

                  {/* Axes dropdown - absolutely positioned so it doesn't affect layout */}
                  {showAdvancedAxes && (isIndoor || isOutdoor) && (
                    <div className="absolute top-full right-0 mt-1 z-50 bg-[#222222] rounded px-2 py-2">
                      <div className="flex items-center gap-2 whitespace-nowrap">
                        {(isIndoor
                          ? ['Objects', 'Lighting', 'Color/Material']
                          : ['Objects', 'Lighting', 'Weather', 'Road Surface']
                        ).map((axis) => (
                          <label
                            key={axis}
                            className="flex items-center gap-1 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              checked={selectedAxes.includes(axis)}
                              onChange={() => {
                                const newAxes = selectedAxes.includes(axis)
                                  ? selectedAxes.filter(a => a !== axis)
                                  : [...selectedAxes, axis]
                                setSelectedAxes(newAxes)
                              }}
                              className="w-3 h-3 accent-[#4b6671] cursor-pointer"
                            />
                            <span className="text-xs text-[#d4d4d4]">{axis}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            {/* Load Dataset Button - 25% width, centered with larger top gap for axes */}
            <div className="flex justify-center mt-10">
              <button
                onClick={handleLoadDataset}
                disabled={uploadLoading || (uploadMode === 'local' && !uploadedFiles) || (uploadMode === 's3' && !s3Path)}
                className="w-1/4 px-3 py-2 text-xs text-white disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-1.5 transition-colors"
                style={{ backgroundColor: '#4b6671' }}
              >
                {uploadLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading...
                  </>
                ) : uploadSuccess ? (
                  <>
                    <CheckCircle2 className="w-4 h-4" />
                    Loaded
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    Load Dataset
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

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
        {currentDataset && !loading && (
          <div className="mb-8">
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Data Analysis</h2>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6 space-y-8">
              <DatasetOverview datasetInfo={datasetInfo} />
              <DatasetDistributions 
                datasetName={currentDataset} 
                aresDistributions={aresDistributions}
              />
              <EpisodePreview datasetData={datasetData} />
            </div>
          </div>
        )}

        {/* Dataset Curation Section */}
        {currentDataset && !loading && (
          <div className="mb-8">
            <h2 className="text-xs font-medium mb-3 text-[#d4d4d4]">Dataset Curation</h2>
            <div className="bg-[#222222] border border-[#2a2a2a] p-6 space-y-6">
              {/* Augmentation Card */}
              <div>
                <h3 className="text-xs font-medium mb-2 text-[#d4d4d4]">Augmentation</h3>
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4">
                  <AugmentationPanel
                    datasetName={currentDataset}
                    onComplete={handleAugmentationComplete}
                  />
                </div>
              </div>

              {/* Optimization Card - Coming Soon */}
              <div>
                <h3 className="text-xs font-medium mb-2 text-[#d4d4d4]">Optimization</h3>
                <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4 opacity-50 pointer-events-none relative">
                  <OptimizationPanel
                    datasetName={currentDataset}
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
                  <TestingPanel datasetName={currentDataset} />
                  <div className="absolute inset-0 flex items-center justify-center bg-[#1a1a1a] bg-opacity-80">
                    <span className="text-sm text-[#9aa4b5] font-medium">Coming Soon</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {!currentDataset && !loading && (
          <div className="text-center py-16">
            <p className="text-[#b5becb] text-xs">
              Load a dataset to get started
            </p>
          </div>
        )}

        {/* Export Section */}
        {currentDataset && !loading && (
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

