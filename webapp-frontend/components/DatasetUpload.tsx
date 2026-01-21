'use client'

import { useState } from 'react'
import { Upload, Loader2, CheckCircle2 } from 'lucide-react'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

interface DatasetUploadProps {
  onDatasetLoaded: (datasetName: string) => void
}

export default function DatasetUpload({ onDatasetLoaded }: DatasetUploadProps) {
  const [datasetPath, setDatasetPath] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)

  const handleLoad = async () => {
    if (!datasetPath || !datasetName) return

    setLoading(true)
    setSuccess(false)

    try {
      const response = await fetch(`${API_BASE}/api/datasets/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_path: datasetPath,
          dataset_name: datasetName,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to load dataset')
      }

      setSuccess(true)
      onDatasetLoaded(datasetName)
    } catch (error: any) {
      alert(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-[#222222] border border-[#2a2a2a] p-4">
      <h2 className="text-xs font-medium mb-3 flex items-center gap-1.5 text-[#e3e8f0]">
        <Upload className="w-3.5 h-3.5 text-[#b5becb]" />
        Upload Data
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        <div className="md:col-span-2">
          <input
            type="text"
            placeholder="path/to/dataset"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666666] text-xs focus:outline-none focus:border-[#3a3a3a] transition-colors"
          />
        </div>
        <div>
          <input
            type="text"
            placeholder="Dataset name"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666666] text-xs focus:outline-none focus:border-[#3a3a3a] transition-colors"
          />
        </div>
        <div className="md:col-span-3">
          <button
            onClick={handleLoad}
            disabled={loading || !datasetPath || !datasetName}
            className="w-full px-3 py-2 text-xs text-white disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-1.5 transition-colors"
            style={{ 
              backgroundColor: '#4b6671',
            }}
            onMouseEnter={(e) => {
              if (!loading && datasetPath && datasetName) {
                e.currentTarget.style.backgroundColor = '#3d5560'
              }
            }}
            onMouseLeave={(e) => {
              if (!loading && datasetPath && datasetName) {
                e.currentTarget.style.backgroundColor = '#4b6671'
              }
            }}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading...
              </>
            ) : success ? (
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
  )
}

