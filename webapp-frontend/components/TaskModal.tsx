'use client'

import { useState } from 'react'
import { X, Loader2 } from 'lucide-react'
import { cn } from '../lib/utils'

interface TaskModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (task: TaskData) => Promise<void>
}

export interface TaskData {
  name: string
  description: string
  dataset_path?: string
  prompt?: string
  priority?: 'low' | 'medium' | 'high'
}

export default function TaskModal({ isOpen, onClose, onSave }: TaskModalProps) {
  const [formData, setFormData] = useState<TaskData>({
    name: '',
    description: '',
    dataset_path: '',
    prompt: '',
    priority: 'medium',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      await onSave(formData)
      // Reset form
      setFormData({
        name: '',
        description: '',
        dataset_path: '',
        prompt: '',
        priority: 'medium',
      })
      onClose()
    } catch (err: any) {
      setError(err.message || 'Failed to save task')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-[#222222] border border-[#2a2a2a] w-full max-w-md">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a]">
          <h2 className="text-sm font-medium text-[#d4d4d4]">Add New Task</h2>
          <button
            onClick={onClose}
            className="p-1 text-[#8a8a8a] hover:text-[#d4d4d4] transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {error && (
            <div className="p-2.5 bg-[#2a1a1a] border border-[#3a2a2a] text-xs text-[#cc6666]">
              {error}
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
              Task Name *
            </label>
            <input
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
              placeholder="Enter task name"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
              Description
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              rows={3}
              className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671] resize-none"
              placeholder="Enter task description"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
              Dataset Path
            </label>
            <input
              type="text"
              value={formData.dataset_path}
              onChange={(e) => setFormData({ ...formData, dataset_path: e.target.value })}
              className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
              placeholder="path/to/dataset"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
              Prompt
            </label>
            <textarea
              value={formData.prompt}
              onChange={(e) => setFormData({ ...formData, prompt: e.target.value })}
              rows={3}
              className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671] resize-none"
              placeholder="Enter prompt text"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
              Priority
            </label>
            <select
              value={formData.priority}
              onChange={(e) => setFormData({ ...formData, priority: e.target.value as 'low' | 'medium' | 'high' })}
              className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] hover:bg-[#2a2a2a] transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 text-xs bg-[#4b6671] text-white hover:bg-[#3d5560] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Saving...
                </>
              ) : (
                'Save Task'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

