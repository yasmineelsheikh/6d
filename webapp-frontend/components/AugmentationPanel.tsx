'use client'

import { useState } from 'react'
import { Play, Loader2, CheckCircle2 } from 'lucide-react'

const API_BASE = 'https://6d-nu.vercel.app'

interface AugmentationPanelProps {
  datasetName: string
  onComplete: (datasetName: string) => void
}

export default function AugmentationPanel({ datasetName, onComplete }: AugmentationPanelProps) {
  const [taskDescription, setTaskDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [s3Url, setS3Url] = useState<string | null>(null)

  const handleRun = async () => {
    setLoading(true)
    setSuccess(false)
    setS3Url(null)

    try {
      const response = await fetch(`${API_BASE}/api/augmentation/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: datasetName,
          prompt: '',
          task_description: taskDescription,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Augmentation failed')
      }

      const result = await response.json()
      setS3Url(result.s3_url)
      setSuccess(true)
      onComplete(datasetName)
    } catch (error: any) {
      alert(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="mb-3">
        <textarea
          value={taskDescription}
          onChange={(e) => setTaskDescription(e.target.value)}
          placeholder="Enter task description"
          rows={2}
          className="w-full px-3 py-1.5 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666666] text-xs focus:outline-none focus:border-[#3a3a3a] resize-none transition-colors"
        />
      </div>
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
    </div>
  )
}

