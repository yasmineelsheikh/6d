'use client'

import { useState } from 'react'
import { X } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

interface RegisterModalProps {
  isOpen: boolean
  onClose: () => void
  onSwitchToLogin: () => void
}

export default function RegisterModal({ isOpen, onClose, onSwitchToLogin }: RegisterModalProps) {
  const [email, setEmail] = useState('')
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { register } = useAuth()

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters')
      return
    }

    if (password.length > 72) {
      setError('Password must be less than 72 characters')
      return
    }

    setLoading(true)

    try {
      await register(email, firstName, lastName, password)
      onClose()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to register')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-[#222222] border border-[#2a2a2a] rounded-lg w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium text-[#d4d4d4]">Register</h2>
          <button
            onClick={onClose}
            className="text-[#8a8a8a] hover:text-[#d4d4d4]"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="p-2 bg-[#2a1a1a] border border-[#3a2a2a] text-[#cc6666] text-xs">
              {error}
            </div>
          )}

          <div>
            <label className="block text-xs text-[#d4d4d4] mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] text-xs focus:outline-none focus:border-[#3a3a3a]"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-[#d4d4d4] mb-1">First Name</label>
              <input
                type="text"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                required
                className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] text-xs focus:outline-none focus:border-[#3a3a3a]"
              />
            </div>
            <div>
              <label className="block text-xs text-[#d4d4d4] mb-1">Last Name</label>
              <input
                type="text"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                required
                className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] text-xs focus:outline-none focus:border-[#3a3a3a]"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs text-[#d4d4d4] mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              maxLength={72}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] text-xs focus:outline-none focus:border-[#3a3a3a]"
            />
          </div>

          <div>
            <label className="block text-xs text-[#d4d4d4] mb-1">Confirm Password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              maxLength={72}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] text-xs focus:outline-none focus:border-[#3a3a3a]"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full px-4 py-2 bg-[#4b6671] text-white text-xs hover:bg-[#3d5560] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Registering...' : 'Register'}
          </button>
        </form>

        <div className="mt-4 text-center">
          <button
            onClick={onSwitchToLogin}
            className="text-xs text-[#8a8a8a] hover:text-[#d4d4d4]"
          >
            Already have an account? Login
          </button>
        </div>
      </div>
    </div>
  )
}
