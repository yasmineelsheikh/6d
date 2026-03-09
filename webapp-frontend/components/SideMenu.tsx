'use client'

import { useState, useEffect } from 'react'
import { Plus, Settings, X, LogOut, CreditCard, Database } from 'lucide-react'
import { useRouter, usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'

interface TaskNavItem {
  name: string
}

interface SideMenuProps {
  onAddTask: () => void
  onOpenSettings: () => void
  onOpenBilling: () => void
  onLogout: () => void
  isOpen: boolean
  onToggle: () => void
  tasks?: TaskNavItem[]
}

export default function SideMenu({ onAddTask, onOpenSettings, onOpenBilling, onLogout, isOpen, onToggle, tasks = [] }: SideMenuProps) {
  const router = useRouter()
  const pathname = usePathname()

  const handleNavigateToTask = (taskName: string) => {
    router.push(`/dataset/${encodeURIComponent(taskName)}`)
    onToggle()
  }

  return (
    <>

      {/* Side Menu Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40"
          onClick={onToggle}
        />
      )}

      {/* Side Menu */}
      <div
        className={cn(
          "fixed left-0 top-0 h-full w-64 bg-[#1e1e1e] border-r border-[#2a2a2a] z-50 transform transition-transform duration-300 ease-in-out flex flex-col",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a]">
          <h2 className="text-sm font-medium text-[#d4d4d4]">6d labs</h2>
          <button
            onClick={onToggle}
            className="p-1 text-[#8a8a8a] hover:text-[#d4d4d4] transition-colors"
            aria-label="Close menu"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* New Task Button */}
        <div className="p-3 border-b border-[#2a2a2a]">
          <button
            onClick={() => {
              onAddTask()
              onToggle()
            }}
            className="w-full flex items-center gap-2.5 px-3 py-2 text-xs font-medium text-white bg-[#4b6671] hover:bg-[#3d5560] transition-colors rounded-lg"
          >
            <Plus className="w-3.5 h-3.5" />
            New Task
          </button>
        </div>

        {/* Tasks List */}
        <div className="flex-1 overflow-y-auto py-2">
          <div className="px-3 mb-2">
            <span className="text-[10px] uppercase tracking-widest text-[#666] font-medium">Tasks</span>
          </div>
          {tasks.length > 0 ? (
            <div className="space-y-0.5 px-2">
              {tasks.map((task) => {
                const isActive = pathname === `/dataset/${encodeURIComponent(task.name)}`
                return (
                  <button
                    key={task.name}
                    onClick={() => handleNavigateToTask(task.name)}
                    className={cn(
                      "w-full flex items-center gap-2.5 px-3 py-2 text-xs transition-colors rounded-lg text-left",
                      isActive
                        ? "bg-white/10 text-white"
                        : "text-[#9aa4b5] hover:text-[#d4d4d4] hover:bg-white/5"
                    )}
                  >
                    <Database className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{task.name}</span>
                  </button>
                )
              })}
            </div>
          ) : (
            <div className="px-5 py-4">
              <p className="text-[11px] text-[#555] italic">No tasks yet</p>
            </div>
          )}
        </div>

        {/* Bottom Section — Account Actions */}
        <div className="border-t border-[#2a2a2a] p-2 space-y-0.5">
          <button
            onClick={() => {
              onOpenSettings()
              onToggle()
            }}
            className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-[#9aa4b5] hover:text-[#d4d4d4] hover:bg-white/5 transition-colors rounded-lg"
          >
            <Settings className="w-3.5 h-3.5" />
            Settings
          </button>

          <button
            onClick={() => {
              onOpenBilling()
              onToggle()
            }}
            className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-[#9aa4b5] hover:text-[#d4d4d4] hover:bg-white/5 transition-colors rounded-lg"
          >
            <CreditCard className="w-3.5 h-3.5" />
            Billing
          </button>

          <button
            onClick={() => {
              onLogout()
              onToggle()
            }}
            className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-[#9aa4b5] hover:text-[#d4d4d4] hover:bg-white/5 transition-colors rounded-lg"
          >
            <LogOut className="w-3.5 h-3.5" />
            Logout
          </button>
        </div>
      </div>
    </>
  )
}
