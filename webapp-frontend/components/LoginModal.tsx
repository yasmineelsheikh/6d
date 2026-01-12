'use client'

import { useState } from 'react'
import { X } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

interface LoginModalProps {
  isOpen: boolean
  onClose: () => void
  onSwitchToRegister: () => void
}

export default function LoginModal({ isOpen, onClose, onSwitchToRegister }: LoginModalProps) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      await login(email, password)
      onClose()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to login')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-[#222222] border border-[#2a2a2a] rounded-lg w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium text-[#d4d4d4]">Login</h2>
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

          <div>
            <label className="block text-xs text-[#d4d4d4] mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] text-xs focus:outline-none focus:border-[#3a3a3a]"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full px-4 py-2 bg-[#4b6671] text-white text-xs hover:bg-[#3d5560] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div className="mt-4 text-center">
          <button
            onClick={onSwitchToRegister}
            className="text-xs text-[#8a8a8a] hover:text-[#d4d4d4]"
          >
            Don't have an account? Register
          </button>
        </div>
      </div>
    </div>
  )
}
