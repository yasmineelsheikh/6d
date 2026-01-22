'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import axios from 'axios'

interface User {
  id: string
  email: string
  first_name: string
  last_name: string
  created_at: string
}

interface AuthContextType {
  user: User | null
  token: string | null
  loading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, first_name: string, last_name: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load from localStorage
    const storedToken = localStorage.getItem('auth_token')
    const storedUser = localStorage.getItem('auth_user')
    
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
      
      // Set up axios interceptor
      axios.defaults.headers.common['Authorization'] = `Bearer ${storedToken}`
    }
    setLoading(false)
  }, [])

  // Set up axios interceptor for all requests
  useEffect(() => {
    const interceptor = axios.interceptors.request.use((config) => {
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      return config
    })

    return () => {
      axios.interceptors.request.eject(interceptor)
    }
  }, [token])

  const login = async (email: string, password: string) => {
    const response = await axios.post(`${API_BASE}/api/auth/login`, { email, password })
    const { access_token, user: userData } = response.data
    
    setToken(access_token)
    setUser(userData)
    localStorage.setItem('auth_token', access_token)
    localStorage.setItem('auth_user', JSON.stringify(userData))
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
  }

  const register = async (email: string, first_name: string, last_name: string, password: string) => {
    const response = await axios.post(`${API_BASE}/api/auth/register`, {
      email,
      first_name,
      last_name,
      password
    })
    const { access_token, user: userData } = response.data
    
    setToken(access_token)
    setUser(userData)
    localStorage.setItem('auth_token', access_token)
    localStorage.setItem('auth_user', JSON.stringify(userData))
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
  }

  const logout = () => {
    setToken(null)
    setUser(null)
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
    delete axios.defaults.headers.common['Authorization']
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
