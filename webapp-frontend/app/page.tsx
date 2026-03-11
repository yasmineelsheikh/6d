'use client'

import { useState, useEffect, useMemo } from 'react'
import { useRouter } from 'next/navigation'
import { Upload, Loader2, CheckCircle2, XCircle, Menu, ChevronDown, ChevronRight, Folder, Cloud, Plus, ArrowRight, TrendingUp } from 'lucide-react'
import DatasetOverview from '@/components/DatasetOverview'
import DatasetDistributions from '@/components/DatasetDistributions'
import EpisodePreview from '@/components/EpisodePreview'
import AugmentationPanel from '@/components/AugmentationPanel'
import OptimizationPanel from '@/components/OptimizationPanel'
import TestingPanel from '@/components/TestingPanel'
import SideMenu from '@/components/SideMenu'
import TaskModal, { TaskData } from '@/components/TaskModal'
import SettingsModal, { SettingsData } from '@/components/SettingsModal'
import BillingModal from '@/components/BillingModal'
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

interface EvaluationSession {
  id: string
  datasetName: string
  uploadedAt: string
  episodeCount: number
  successRate: number
}

interface EvaluationTask {
  id: string
  name: string
  sessions: EvaluationSession[]
  isExpanded: boolean
}

const SEED_TASKS: EvaluationTask[] = [
  {
    id: 'task-1',
    name: 'Stack Cups',
    isExpanded: false,
    sessions: [
      { id: 's1', datasetName: 'stack_cups_v1', uploadedAt: '2026-01-15', episodeCount: 80, successRate: 88 },
      { id: 's2', datasetName: 'stack_cups_v2', uploadedAt: '2026-02-03', episodeCount: 120, successRate: 87 },
    ],
  },
  {
    id: 'task-2',
    name: 'Fold Laundry',
    isExpanded: false,
    sessions: [
      { id: 's3', datasetName: 'laundry_indoor_jan', uploadedAt: '2026-01-20', episodeCount: 45, successRate: 72 },
    ],
  },
  {
    id: 'task-3',
    name: 'Open Door',
    isExpanded: false,
    sessions: [
      { id: 's4', datasetName: 'door_handle_dataset', uploadedAt: '2026-01-28', episodeCount: 60, successRate: 91 },
    ],
  },
  {
    id: 'task-4',
    name: 'Pour Liquid',
    isExpanded: false,
    sessions: [],
  },
  {
    id: 'task-5',
    name: 'Pick and Place',
    isExpanded: false,
    sessions: [
      { id: 's5', datasetName: 'pick_place_objects', uploadedAt: '2026-02-10', episodeCount: 95, successRate: 83 },
    ],
  },
]

// ── Sparkline Component ──
function Sparkline({ values, width = 80, height = 28 }: { values: number[]; width?: number; height?: number }) {
  if (values.length < 2) return null
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width
      const y = height - ((v - min) / range) * (height - 4) - 2
      return `${x},${y}`
    })
    .join(' ')

  const lastValue = values[values.length - 1]
  const prevValue = values[values.length - 2]
  const color = lastValue >= prevValue ? '#5fa35f' : '#cc6666'

  return (
    <svg width={width} height={height} className="flex-shrink-0">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

// ── Status dot helper ──
function StatusDot({ rate }: { rate: number | null }) {
  if (rate === null) return <span className="w-2 h-2 rounded-full bg-[#555] flex-shrink-0" />
  const color = rate >= 85 ? 'bg-[#5fa35f]' : rate >= 70 ? 'bg-[#c0a854]' : 'bg-[#cc6666]'
  return <span className={`w-2 h-2 rounded-full ${color} flex-shrink-0`} />
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
  const [uploadMode, setUploadMode] = useState<'local' | 's3' | 'huggingface'>('local')
  const [uploadedFiles, setUploadedFiles] = useState<FileList | null>(null)

  // S3 credentials state
  const [s3AccessKey, setS3AccessKey] = useState('')
  const [s3SecretKey, setS3SecretKey] = useState('')
  const [s3Bucket, setS3Bucket] = useState('')
  const [s3Region, setS3Region] = useState('')
  const [s3UserPath, setS3UserPath] = useState('')

  // Hugging Face state
  const [hfRepoId, setHfRepoId] = useState('')
  const [hfSplit, setHfSplit] = useState('train')
  const [hfToken, setHfToken] = useState('')
  const [ingestionStatus, setIngestionStatus] = useState<'in_progress' | 'complete' | 'failed' | null>(null)
  const [ingestingDatasetName, setIngestingDatasetName] = useState<string | null>(null)
  const [ingestionProgress, setIngestionProgress] = useState(0)

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
  const [isBillingModalOpen, setIsBillingModalOpen] = useState(false)
  const [tasks, setTasks] = useState<TaskData[]>([])

  // Evaluation tab state
  const [evaluationTasks, setEvaluationTasks] = useState<EvaluationTask[]>(SEED_TASKS)
  const [isUploadSectionExpanded, setIsUploadSectionExpanded] = useState(true)
  const [evalSelectedTask, setEvalSelectedTask] = useState<string>('Stack Cups')
  const [evalNewTaskName, setEvalNewTaskName] = useState('')


  // Debug: verify API base URL at runtime
  useEffect(() => {
    console.log('API_BASE at runtime:', API_BASE)
  }, [])

  const handleEvaluationUpload = async () => {
    if (uploadMode === 'local' && !uploadedFiles) return
    if (uploadMode === 's3' && (!s3AccessKey || !s3SecretKey || !s3Bucket || !s3UserPath)) return
    if (uploadMode === 'huggingface' && !hfRepoId) return

    // Determine task name
    const taskName = evalSelectedTask === '__new__' ? evalNewTaskName.trim() : evalSelectedTask
    if (!taskName) return

    // Auto-generate dataset name if not set
    let finalDatasetName = datasetName
    if (!finalDatasetName) {
      if (uploadMode === 'local' && uploadedFiles) {
        const firstFile = uploadedFiles[0]
        const relativePath = (firstFile as any).webkitRelativePath || firstFile.name
        finalDatasetName = relativePath.split('/')[0]
      } else if (uploadMode === 's3' && s3UserPath) {
        const pathParts = s3UserPath.split('/').filter(p => p)
        finalDatasetName = pathParts[pathParts.length - 1] || 'dataset'
      } else if (uploadMode === 'huggingface' && hfRepoId) {
        finalDatasetName = hfRepoId.split('/').pop() || 'dataset'
      } else {
        return
      }
      setDatasetName(finalDatasetName)
    }

    setUploadLoading(true)
    setUploadSuccess(false)
    setError(null)

    // Simulate upload processing
    setTimeout(() => {
      const newSession: EvaluationSession = {
        id: `s-${Date.now()}`,
        datasetName: finalDatasetName,
        uploadedAt: new Date().toISOString().split('T')[0],
        episodeCount: Math.floor(Math.random() * 80) + 40,
        successRate: Math.floor(Math.random() * 25) + 70,
      }

      setEvaluationTasks(prev => {
        const existing = prev.find(t => t.name === taskName)
        if (existing) {
          // Update existing task — append session and auto-expand
          return prev.map(t =>
            t.name === taskName
              ? { ...t, sessions: [...t.sessions, newSession], isExpanded: true }
              : { ...t, isExpanded: false }
          )
        } else {
          // Create new task
          return [
            ...prev.map(t => ({ ...t, isExpanded: false })),
            {
              id: `task-${Date.now()}`,
              name: taskName,
              sessions: [newSession],
              isExpanded: true,
            },
          ]
        }
      })

      setUploadLoading(false)
      setUploadSuccess(true)
      setIsUploadSectionExpanded(false)
      setEvalNewTaskName('')
    }, 1500)
  }

  const handleLoadDataset = async () => {
    if (uploadMode === 'local' && !uploadedFiles) return
    if (uploadMode === 's3' && (!s3AccessKey || !s3SecretKey || !s3Bucket || !s3UserPath)) return
    if (uploadMode === 'huggingface' && !hfRepoId) return

    // Auto-generate dataset name if not set
    let finalDatasetName = datasetName
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
      } else if (uploadMode === 's3') {
        // User's S3 bucket with credentials
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
        // Hugging Face dataset
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

      // Upload successful - start polling for ingestion status
      setUploadSuccess(true)
      setIngestingDatasetName(finalDatasetName)
      setIngestionStatus('in_progress')
      setIngestionProgress(10) // Start at 10%

      // Poll for ingestion completion
      const pollIngestionStatus = async () => {
        const maxAttempts = 300 // 5 minutes max (300 * 1 second)
        let attempts = 0

        const checkStatus = async (): Promise<void> => {
          try {
            const statusResponse = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(finalDatasetName)}/ingestion-status`)
            if (statusResponse.ok) {
              const statusData = await statusResponse.json()
              const status = statusData.status

              if (status === 'complete') {
                setIngestionProgress(100)
                setIngestionStatus('complete')
                // Small delay to show 100% progress, then navigate
                setTimeout(() => {
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
                  setUploadLoading(false)
                }, 500)
                return
              } else if (status === 'failed') {
                setIngestionStatus('failed')
                setError('Dataset ingestion failed. Please try again.')
                setUploadLoading(false)
                return
              } else if (status === 'in_progress') {
                setIngestionStatus('in_progress')
                // Simulate progress (0-90%) while in progress
                const progress = Math.min(90, 10 + (attempts * 0.5))
                setIngestionProgress(progress)
                attempts++
                if (attempts < maxAttempts) {
                  setTimeout(checkStatus, 1000) // Poll every 1 second
                } else {
                  setError('Ingestion is taking longer than expected. Please check back later.')
                  setUploadLoading(false)
                }
              } else {
                // not_started - wait a bit and check again
                attempts++
                if (attempts < maxAttempts) {
                  setTimeout(checkStatus, 1000)
                } else {
                  setError('Ingestion did not start. Please try again.')
                  setUploadLoading(false)
                }
              }
            } else {
              // Status endpoint error - assume in progress and keep polling
              attempts++
              if (attempts < maxAttempts) {
                setTimeout(checkStatus, 1000)
              } else {
                setError('Could not check ingestion status. Please try again.')
                setUploadLoading(false)
              }
            }
          } catch (err) {
            console.error('Error checking ingestion status:', err)
            attempts++
            if (attempts < maxAttempts) {
              setTimeout(checkStatus, 1000)
            } else {
              setError('Error checking ingestion status. Please try again.')
              setUploadLoading(false)
            }
          }
        }

        // Start polling after a short delay
        setTimeout(checkStatus, 1000)
      }

      pollIngestionStatus()
    } catch (error: any) {
      setError(error.message)
      setUploadLoading(false)
      setIngestionStatus(null)
      setIngestingDatasetName(null)
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
      const response = await fetch(`${API_BASE}/api/datasets/${datasetName}/info`)
      if (!response.ok) throw new Error('Failed to load dataset info')
      const info = await response.json()
      setDatasetInfo(info)

      const dataResponse = await fetch(`${API_BASE}/api/datasets/${datasetName}/data`)
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

        const url = `${API_BASE}/api/ares/distributions${params.toString() ? '?' + params.toString() : ''}`
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
      const response = await fetch(`${API_BASE}/api/datasets/${currentDataset}/export`)
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Failed to export dataset')
      }

      // Get the zip file as a blob
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${currentDataset}_export.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (err: any) {
      console.error('Export error:', err)
      alert(err.message || 'Failed to export dataset')
    }
  }

  // Show login modal if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      setIsLoginModalOpen(true)
    }
  }, [authLoading, isAuthenticated])

  // ── Derived dashboard data ──
  const sortedTasks = useMemo(() => {
    return [...evaluationTasks].sort((a, b) => {
      // Flagged tasks first (no sessions, or declining rate)
      const aFlagged = a.sessions.length === 0 || (a.sessions.length >= 2 && a.sessions[a.sessions.length - 1].successRate < a.sessions[a.sessions.length - 2].successRate)
      const bFlagged = b.sessions.length === 0 || (b.sessions.length >= 2 && b.sessions[b.sessions.length - 1].successRate < b.sessions[b.sessions.length - 2].successRate)
      if (aFlagged && !bFlagged) return -1
      if (!aFlagged && bFlagged) return 1
      // Then by most recent uploadedAt
      const aLatest = a.sessions.length > 0 ? a.sessions[a.sessions.length - 1].uploadedAt : ''
      const bLatest = b.sessions.length > 0 ? b.sessions[b.sessions.length - 1].uploadedAt : ''
      return bLatest.localeCompare(aLatest)
    })
  }, [evaluationTasks])

  const fleetStats = useMemo(() => {
    const total = evaluationTasks.length
    return { total }
  }, [evaluationTasks])


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
    <div className="min-h-screen bg-[#1a1a1a] text-white">
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
        onOpenBilling={() => setIsBillingModalOpen(true)}
        onLogout={() => {
          logout()
          setIsLoginModalOpen(true)
          setIsRegisterModalOpen(false)
          router.push('/')
        }}
        tasks={evaluationTasks.map(t => ({ name: t.name }))}
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

      {/* Billing Modal */}
      <BillingModal
        isOpen={isBillingModalOpen}
        onClose={() => setIsBillingModalOpen(false)}
      />

      {/* Header */}
      <header className="border-b border-[#2a2a2a] bg-[#222222] sticky top-0 z-50">
        <div className="flex items-center justify-between h-10">
          <div className="flex items-center h-full">
            <button
              onClick={() => setIsSideMenuOpen(!isSideMenuOpen)}
              className="px-4 h-full text-white/40 hover:text-white hover:bg-[#2a2a2a] transition-colors border-r border-[#2a2a2a]"
              aria-label="Toggle menu"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-4 px-6 flex-1">
              <h1 className="text-sm font-medium tracking-wide text-white">
                6d labs
              </h1>
            </div>
          </div>
          <div className="px-4">
            <button
              onClick={() => setIsTaskModalOpen(true)}
              className="flex items-center gap-1.5 px-4 py-2 text-[11px] font-semibold tracking-wide uppercase text-white bg-gradient-to-r from-[#4b6671] to-[#3d5f6f] hover:from-[#567a86] hover:to-[#4b6f7f] transition-all rounded-xl shadow-lg shadow-[#4b6671]/20 hover:shadow-[#4b6671]/30"
            >
              <Plus className="w-3.5 h-3.5" />
              New Task
            </button>
          </div>
        </div>
      </header>

      {/* Show loading page while ingestion is in progress */}
      {ingestionStatus === 'in_progress' && ingestingDatasetName ? (
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
      ) : (
        <>
          {/* Fleet Health Bar */}
          <div className="border-b border-[#2a2a2a] bg-[#1e1e1e]">
            <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-8">
              <div className="flex items-center gap-2">
                <span className="text-[10px] uppercase tracking-widest text-white/30 font-medium">Tasks</span>
                <span className="text-sm font-medium text-white">{fleetStats.total}</span>
              </div>
            </div>
          </div>

          <main className="max-w-7xl mx-auto px-6 py-8">
            {/* Task Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {sortedTasks.map(task => {
                const totalEpisodes = task.sessions.reduce((sum, s) => sum + s.episodeCount, 0)
                const avgSuccess = task.sessions.length > 0
                  ? Math.round(task.sessions.reduce((sum, s) => sum + s.successRate, 0) / task.sessions.length)
                  : null
                const successRates = task.sessions.map(s => s.successRate)
                const lastUpdated = task.sessions.length > 0
                  ? task.sessions[task.sessions.length - 1].uploadedAt
                  : null


                return (
                  <button
                    key={task.id}
                    onClick={() => router.push(`/dataset/${encodeURIComponent(task.name)}`)}
                    className="bg-white/5 border border-white/10 rounded-xl p-5 text-left hover:bg-white/[0.07] hover:border-white/[0.15] transition-all group"
                  >
                    {/* Card header */}
                    <div className="flex items-center gap-2.5 mb-4">
                      <StatusDot rate={avgSuccess} />
                      <h3 className="text-sm font-medium text-white group-hover:text-white/90">{task.name}</h3>
                    </div>

                    {/* Sparkline */}
                    <div className="mb-4 flex items-center gap-3">
                      {successRates.length >= 2 ? (
                        <>
                          <Sparkline values={successRates} width={100} height={28} />
                          <span className={cn(
                            "text-xs font-mono",
                            avgSuccess !== null && avgSuccess >= 85 ? 'text-[#5fa35f]' :
                              avgSuccess !== null && avgSuccess >= 70 ? 'text-[#c0a854]' : 'text-[#cc6666]'
                          )}>
                            {avgSuccess}%
                          </span>
                        </>
                      ) : avgSuccess !== null ? (
                        <span className={cn(
                          "text-xs font-mono",
                          avgSuccess >= 85 ? 'text-[#5fa35f]' :
                            avgSuccess >= 70 ? 'text-[#c0a854]' : 'text-[#cc6666]'
                        )}>
                          {avgSuccess}% avg
                        </span>
                      ) : (
                        <span className="text-[11px] text-white/30 italic">No data</span>
                      )}
                    </div>

                    {/* Health indicators */}
                    <div className="flex items-center gap-4 text-[11px] text-white/40">
                      <span>{totalEpisodes} samples</span>
                      <span className="text-[#2a2a2a]">·</span>
                      <span>{task.sessions.length} sessions</span>
                      {lastUpdated && (
                        <>
                          <span className="text-[#2a2a2a]">·</span>
                          <span>{lastUpdated}</span>
                        </>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          </main >
        </>
      )
      }

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
    </div >
  )
}
