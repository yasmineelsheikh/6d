'use client'

import { useState } from 'react'
import { Play, Loader2, CheckCircle2 } from 'lucide-react'

const API_BASE = 'https://6d-nu.vercel.app'

interface TestingPanelProps {
  datasetName: string
}

export default function TestingPanel({ datasetName }: TestingPanelProps) {
  const [testDirectory, setTestDirectory] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleUpload = async () => {
    if (!testDirectory.trim()) {
      setError('Please enter a directory path')
      return
    }

    setLoading(true)
    setSuccess(false)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/api/testing/upload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: datasetName,
          test_directory: testDirectory,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const result = await response.json()
      setSuccess(true)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="mb-3">
        <input
          type="text"
          value={testDirectory}
          onChange={(e) => setTestDirectory(e.target.value)}
          placeholder="path/to/test/data"
          className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666666] text-xs focus:outline-none focus:border-[#3a3a3a] transition-colors"
        />
      </div>
        <button
          onClick={handleUpload}
          disabled={loading || !testDirectory.trim()}
          className="w-full px-3 py-2 text-xs text-white disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center gap-1.5 transition-colors"
          style={{ backgroundColor: '#4b6671' }}
        >
          {loading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Running...
            </>
          ) : success ? (
            <>
              <CheckCircle2 className="w-3.5 h-3.5" />
              Uploaded
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" />
              Execute
            </>
          )}
        </button>
        {error && (
          <div className="mt-3 p-2.5 bg-[#2a1a1a] border border-[#3a2a2a]">
            <p className="text-xs text-[#cc6666]">{error}</p>
          </div>
        )}
        {success && (
          <div className="mt-3 p-2.5 bg-[#1a2a1a] border border-[#2a3a2a]">
            <p className="text-xs text-[#88aa88]">
              Test data updated successfully
            </p>
          </div>
        )}
    </div>
  )
}

