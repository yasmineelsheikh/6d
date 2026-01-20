'use client'

import { useState, useEffect } from 'react'
import { Plus, Settings, X, LogOut } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SideMenuProps {
  onAddTask: () => void
  onOpenSettings: () => void
  onLogout: () => void
  isOpen: boolean
  onToggle: () => void
}

export default function SideMenu({ onAddTask, onOpenSettings, onLogout, isOpen, onToggle }: SideMenuProps) {

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
          "fixed left-0 top-0 h-full w-64 bg-[#222222] border-r border-[#2a2a2a] z-50 transform transition-transform duration-300 ease-in-out",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-[#2a2a2a]">
            <h2 className="text-sm font-medium text-[#d4d4d4]">Menu</h2>
            <button
              onClick={onToggle}
              className="p-1 text-[#8a8a8a] hover:text-[#d4d4d4] transition-colors"
              aria-label="Close menu"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Menu Items */}
          <div className="flex-1 p-4 space-y-2">
            <button
              onClick={() => {
                onAddTask()
                onToggle()
              }}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm font-medium text-[#d4d4d4] bg-[#1a1a1a] border border-[#2a2a2a] hover:bg-[#2a2a2a] transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Task
            </button>

            <button
              onClick={() => {
                onOpenSettings()
                onToggle()
              }}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm font-medium text-[#d4d4d4] hover:text-[#e3e8f0] transition-colors"
            >
              <Settings className="w-4 h-4" />
              Settings
            </button>

            <button
              onClick={() => {
                onLogout()
                onToggle()
              }}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm font-medium text-[#d4d4d4] hover:text-[#e3e8f0] transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      </div>
    </>
  )
}

