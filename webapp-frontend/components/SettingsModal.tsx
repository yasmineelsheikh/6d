'use client'

import { useState, useEffect } from 'react'
import { X, Loader2, Save } from 'lucide-react'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (settings: SettingsData) => Promise<void>
}

export interface SettingsData {
  theme?: 'dark' | 'light'
}

export default function SettingsModal({ isOpen, onClose, onSave }: SettingsModalProps) {
  const [formData, setFormData] = useState<SettingsData>({
    theme: 'dark',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  // Load settings from localStorage on mount
  useEffect(() => {
    if (isOpen) {
      const savedSettings = localStorage.getItem('app_settings')
      if (savedSettings) {
        try {
          const parsed = JSON.parse(savedSettings)
          setFormData({ ...formData, ...parsed })
        } catch (e) {
          console.error('Failed to parse saved settings:', e)
        }
      }
    }
  }, [isOpen])

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSaved(false)

    try {
      // Save to localStorage
      localStorage.setItem('app_settings', JSON.stringify(formData))
      
      await onSave(formData)
      setSaved(true)
      setTimeout(() => {
        setSaved(false)
        onClose()
      }, 1000)
    } catch (err: any) {
      setError(err.message || 'Failed to save settings')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-[#222222] border border-[#2a2a2a] w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a] sticky top-0 bg-[#222222] z-10">
          <h2 className="text-sm font-medium text-[#d4d4d4]">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 text-[#8a8a8a] hover:text-[#d4d4d4] transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-6">
          {error && (
            <div className="p-2.5 bg-[#2a1a1a] border border-[#3a2a2a] text-xs text-[#cc6666]">
              {error}
            </div>
          )}

          {saved && (
            <div className="p-2.5 bg-[#1a2a1a] border border-[#2a3a2a] text-xs text-[#66cc66]">
              Settings saved
            </div>
          )}

          

          {/* Display Settings */}
          <div className="space-y-4">

            <div>
              <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
                Theme
              </label>
              <select
                value={formData.theme}
                onChange={(e) => setFormData({ ...formData, theme: e.target.value as 'dark' | 'light' })}
                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
              >
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-4 border-t border-[#2a2a2a]">
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
                <>
                  Save
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

