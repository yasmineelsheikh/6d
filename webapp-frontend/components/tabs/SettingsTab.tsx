'use client'

import { useState } from 'react'
import { Loader2, Save, Lock, Mail } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

export default function SettingsTab() {
    const { user, token } = useAuth()
    const [activeSection, setActiveSection] = useState<'password' | 'email'>('password')

    const [currentPassword, setCurrentPassword] = useState('')
    const [newPassword, setNewPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [newEmail, setNewEmail] = useState('')
    const [emailPassword, setEmailPassword] = useState('')

    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)

    const handlePasswordChange = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true); setError(null); setSuccess(null)
        if (!currentPassword || !newPassword || !confirmPassword) { setError('All fields are required'); setLoading(false); return }
        if (newPassword !== confirmPassword) { setError('New passwords do not match'); setLoading(false); return }
        if (newPassword.length < 6) { setError('New password must be at least 6 characters'); setLoading(false); return }
        try {
            const response = await fetch(`${API_BASE}/api/auth/change-password`, {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
            })
            if (!response.ok) { const d = await response.json(); throw new Error(d.detail || 'Failed') }
            setSuccess('Password changed successfully')
            setCurrentPassword(''); setNewPassword(''); setConfirmPassword('')
        } catch (err: any) { setError(err.message) } finally { setLoading(false) }
    }

    const handleEmailChange = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true); setError(null); setSuccess(null)
        if (!newEmail || !emailPassword) { setError('All fields are required'); setLoading(false); return }
        if (newEmail === user?.email) { setError('Must be different from current email'); setLoading(false); return }
        try {
            const response = await fetch(`${API_BASE}/api/auth/change-email`, {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify({ new_email: newEmail, password: emailPassword })
            })
            if (!response.ok) { const d = await response.json(); throw new Error(d.detail || 'Failed') }
            setSuccess('Email changed successfully')
            setNewEmail(''); setEmailPassword('')
            if (user) { user.email = (await response.json()).new_email }
        } catch (err: any) { setError(err.message) } finally { setLoading(false) }
    }

    return (
        <div className="max-w-xl space-y-6">
            {/* Section tabs */}
            <div className="flex gap-4 border-b border-white/10 pb-px">
                <button onClick={() => { setActiveSection('password'); setError(null); setSuccess(null) }}
                    className={`pb-2 text-xs font-medium flex items-center gap-1.5 border-b-2 transition-colors ${activeSection === 'password' ? 'border-[#4b6671] text-white' : 'border-transparent text-[#9aa4b5] hover:text-[#d4d4d4]'}`}>
                    <Lock className="w-3.5 h-3.5" /> Reset Password
                </button>
                <button onClick={() => { setActiveSection('email'); setError(null); setSuccess(null) }}
                    className={`pb-2 text-xs font-medium flex items-center gap-1.5 border-b-2 transition-colors ${activeSection === 'email' ? 'border-[#4b6671] text-white' : 'border-transparent text-[#9aa4b5] hover:text-[#d4d4d4]'}`}>
                    <Mail className="w-3.5 h-3.5" /> Change Email
                </button>
            </div>

            {error && <div className="p-2.5 bg-red-500/10 border border-red-500/20 text-xs text-red-400 rounded-lg">{error}</div>}
            {success && <div className="p-2.5 bg-green-500/10 border border-green-500/20 text-xs text-green-400 rounded-lg">{success}</div>}

            <form onSubmit={activeSection === 'password' ? handlePasswordChange : handleEmailChange} className="space-y-4">
                {activeSection === 'password' && (
                    <>
                        <div>
                            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">Current Password</label>
                            <input type="password" value={currentPassword} onChange={e => setCurrentPassword(e.target.value)} required
                                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] rounded-lg focus:outline-none focus:border-[#4b6671]" placeholder="Enter current password" />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">New Password</label>
                            <input type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)} required minLength={6}
                                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] rounded-lg focus:outline-none focus:border-[#4b6671]" placeholder="Min 6 characters" />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">Confirm New Password</label>
                            <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} required minLength={6}
                                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] rounded-lg focus:outline-none focus:border-[#4b6671]" placeholder="Confirm new password" />
                        </div>
                    </>
                )}
                {activeSection === 'email' && (
                    <>
                        <div>
                            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">Current Email</label>
                            <input type="email" value={user?.email || ''} disabled className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#666] rounded-lg cursor-not-allowed" />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">New Email</label>
                            <input type="email" value={newEmail} onChange={e => setNewEmail(e.target.value)} required
                                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] rounded-lg focus:outline-none focus:border-[#4b6671]" placeholder="Enter new email" />
                        </div>
                        <div>
                            <label className="block text-xs font-medium text-[#d4d4d4] mb-1.5">Confirm Password</label>
                            <input type="password" value={emailPassword} onChange={e => setEmailPassword(e.target.value)} required
                                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] rounded-lg focus:outline-none focus:border-[#4b6671]" placeholder="Enter password to confirm" />
                        </div>
                    </>
                )}
                <button type="submit" disabled={loading}
                    className="px-4 py-2 text-xs bg-[#4b6671] text-white hover:bg-[#3d5560] transition-colors rounded-lg disabled:opacity-50 flex items-center gap-2">
                    {loading ? <><Loader2 className="w-3 h-3 animate-spin" /> Saving...</> : <><Save className="w-3 h-3" /> {activeSection === 'password' ? 'Change Password' : 'Change Email'}</>}
                </button>
            </form>

            {/* Placeholder sections */}
            <div className="space-y-4 pt-4 border-t border-white/10">
                <div>
                    <span className="text-[10px] uppercase tracking-widest text-[#666] font-medium">Augmentation Preferences</span>
                    <div className="mt-2 bg-white/5 border border-white/10 rounded-xl p-4">
                        <p className="text-[11px] text-[#555] italic">Coming soon</p>
                    </div>
                </div>
                <div>
                    <span className="text-[10px] uppercase tracking-widest text-[#666] font-medium">Alert Thresholds</span>
                    <div className="mt-2 bg-white/5 border border-white/10 rounded-xl p-4">
                        <p className="text-[11px] text-[#555] italic">Coming soon</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
