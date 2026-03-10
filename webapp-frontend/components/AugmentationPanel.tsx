'use client'

import { useState } from 'react'
import { Play, Loader2, CheckCircle2, Folder, Cloud } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

interface AugmentationPanelProps {
  datasetName: string
  onComplete: (datasetName: string) => void
}

export default function AugmentationPanel({ datasetName, onComplete }: AugmentationPanelProps) {
  const { token } = useAuth()
  const [taskDescription, setTaskDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [s3Url, setS3Url] = useState<string | null>(null)
  const [exportLocation, setExportLocation] = useState<string | null>(null)

  // Export destination state
  const [exportMode, setExportMode] = useState<'local' | 's3' | null>(null)
  const [localPath, setLocalPath] = useState('')
  const [s3Bucket, setS3Bucket] = useState('')
  const [s3Region, setS3Region] = useState('')
  const [s3Path, setS3Path] = useState('')
  const [s3AccessKey, setS3AccessKey] = useState('')
  const [s3SecretKey, setS3SecretKey] = useState('')

  const handleRun = async () => {
    setLoading(true)
    setSuccess(false)
    setS3Url(null)
    setExportLocation(null)

    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      }

      if (token) {
        headers['Authorization'] = `Bearer ${token}`
      }

      const response = await fetch(`${API_BASE}/api/augmentation/run`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          dataset_name: datasetName,
          prompt: '',
          task_description: taskDescription,
          export_mode: exportMode,
          local_path: exportMode === 'local' ? localPath : undefined,
          s3_bucket: exportMode === 's3' ? s3Bucket : undefined,
          s3_region: exportMode === 's3' ? s3Region : undefined,
          s3_path: exportMode === 's3' ? s3Path : undefined,
          s3_access_key: exportMode === 's3' ? s3AccessKey : undefined,
          s3_secret_key: exportMode === 's3' ? s3SecretKey : undefined,
        }),
      })

      if (!response.ok) {
        let errorMessage = 'Augmentation failed'
        try {
          const error = await response.json()
          if (error?.detail) {
            errorMessage = error.detail
          }
        } catch {
          // ignore JSON parse errors
        }
        throw new Error(errorMessage)
      }

      const result = await response.json()
      setS3Url(result.s3_url)
      setExportLocation(result.export_location)
      setSuccess(true)
      onComplete(datasetName)
    } catch (error: any) {
      alert(error.message || 'Augmentation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <button
        onClick={handleRun}
        disabled={loading}
        className="w-full px-3 py-2 text-xs text-white disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-1.5 transition-colors"
        style={{ backgroundColor: '#4b6671' }}
      >
        {loading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Running augmentation...
          </>
        ) : success ? (
          <>
            <CheckCircle2 className="w-3.5 h-3.5" />
            Complete
          </>
        ) : (
          <>
            <Play className="w-3.5 h-3.5" />
            Execute
          </>
        )}
      </button>
      {s3Url && (
        <div className="mt-3 p-2.5 bg-[#1a2a1a] border border-[#2a3a2a]">
          <p className="text-xs text-[#88aa88] mb-1">
            Output video uploaded to S3:
          </p>
          <p className="text-xs text-[#88aa88] font-mono break-all">{s3Url}</p>
        </div>
      )}
      {exportLocation && (
        <div className="mt-3 p-2.5 bg-[#1a2a3a] border border-[#2a3a4a]">
          <p className="text-xs text-[#88aacc] mb-1">
            Exported to:
          </p>
          <p className="text-xs text-[#88aacc] font-mono break-all">{exportLocation}</p>
        </div>
      )}
    </div>
  )
}

