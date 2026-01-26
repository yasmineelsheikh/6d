'use client'

import { useState } from 'react'
import { X, Loader2, Save, Lock, Mail } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (settings: SettingsData) => Promise<void>
}

export interface SettingsData {
  // No longer using theme
}

export default function SettingsModal({ isOpen, onClose, onSave }: SettingsModalProps) {
  const { user, token } = useAuth()
  const [activeTab, setActiveTab] = useState<'password' | 'email'>('password')
  
  // Password change form
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  
  // Email change form
  const [newEmail, setNewEmail] = useState('')
  const [emailPassword, setEmailPassword] = useState('')
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    // Validation
    if (!currentPassword || !newPassword || !confirmPassword) {
      setError('All fields are required')
      setLoading(false)
      return
    }

    if (newPassword !== confirmPassword) {
      setError('New passwords do not match')
      setLoading(false)
      return
    }

    if (newPassword.length < 6) {
      setError('New password must be at least 6 characters long')
      setLoading(false)
      return
    }

    try {
      const response = await fetch(`${API_BASE}/api/auth/change-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to change password')
      }

      setSuccess('Password changed successfully')
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      
      setTimeout(() => {
        setSuccess(null)
        onClose()
      }, 2000)
    } catch (err: any) {
      setError(err.message || 'Failed to change password')
    } finally {
      setLoading(false)
    }
  }

  const handleEmailChange = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    // Validation
    if (!newEmail || !emailPassword) {
      setError('All fields are required')
      setLoading(false)
      return
    }

    if (newEmail === user?.email) {
      setError('New email must be different from current email')
      setLoading(false)
      return
    }

    try {
      const response = await fetch(`${API_BASE}/api/auth/change-email`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          new_email: newEmail,
          password: emailPassword
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to change email')
      }

      const data = await response.json()
      setSuccess('Email changed successfully')
      setNewEmail('')
      setEmailPassword('')
      
      // Update user in context
      if (user) {
        user.email = data.new_email
      }
      
      setTimeout(() => {
        setSuccess(null)
        onClose()
      }, 2000)
    } catch (err: any) {
      setError(err.message || 'Failed to change email')
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

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

        {/* Tabs */}
        <div className="flex border-b border-[#2a2a2a]">
          <button
            onClick={() => {
              setActiveTab('password')
              setError(null)
              setSuccess(null)
            }}
            className={`flex-1 px-4 py-3 text-xs font-medium transition-colors flex items-center justify-center gap-2 ${
              activeTab === 'password'
                ? 'text-[#d4d4d4] border-b-2 border-[#4b6671]'
                : 'text-[#8a8a8a] hover:text-[#d4d4d4]'
            }`}
          >
            <Lock className="w-3.5 h-3.5" />
            Reset Password
          </button>
          <button
            onClick={() => {
              setActiveTab('email')
              setError(null)
              setSuccess(null)
            }}
            className={`flex-1 px-4 py-3 text-xs font-medium transition-colors flex items-center justify-center gap-2 ${
              activeTab === 'email'
                ? 'text-[#d4d4d4] border-b-2 border-[#4b6671]'
                : 'text-[#8a8a8a] hover:text-[#d4d4d4]'
            }`}
          >
            <Mail className="w-3.5 h-3.5" />
            Change Email
          </button>
        </div>

        {/* Form */}
        <form 
          onSubmit={activeTab === 'password' ? handlePasswordChange : handleEmailChange}
          className="p-4 space-y-6"
        >
          {error && (
            <div className="p-2.5 bg-red-500/10 border border-red-500/20 text-xs text-red-400 rounded">
              {error}
            </div>
          )}

          {success && (
            <div className="p-2.5 bg-green-500/10 border border-green-500/20 text-xs text-green-400 rounded">
              {success}
            </div>
          )}

          {/* Password Change Form */}
          {activeTab === 'password' && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
                  Current Password
                </label>
                <input
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
                  placeholder="Enter current password"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
                  New Password
                </label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
                  placeholder="Enter new password (min 6 characters)"
                  required
                  minLength={6}
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
                  Confirm New Password
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
                  placeholder="Confirm new password"
                  required
                  minLength={6}
                />
              </div>
            </div>
          )}

          {/* Email Change Form */}
          {activeTab === 'email' && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
                  Current Email
                </label>
                <input
                  type="email"
                  value={user?.email || ''}
                  disabled
                  className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#666666] cursor-not-allowed"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
                  New Email
                </label>
                <input
                  type="email"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
                  placeholder="Enter new email address"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">
                  Confirm Password
                </label>
                <input
                  type="password"
                  value={emailPassword}
                  onChange={(e) => setEmailPassword(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] focus:outline-none focus:border-[#4b6671]"
                  placeholder="Enter your password to confirm"
                  required
                />
                <p className="text-xs text-[#8a8a8a] mt-1">
                  Please enter your password to confirm the email change
                </p>
              </div>
            </div>
          )}

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
                  {activeTab === 'password' ? 'Changing...' : 'Updating...'}
                </>
              ) : (
                <>
                  <Save className="w-3 h-3" />
                  {activeTab === 'password' ? 'Change Password' : 'Change Email'}
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
